# Generates code with two Hugging Face models and three approaches (baseline, BANDIT-REFINE, SELF-FEEDBACK)
# for every prompt of the RiskyPy dataset, saved in runs/<model>__<approach>.json, and for the prompts the
# pipeline dropped at the func check or at the vuln check, saved in runs/dropped_prompts_runs/
# <model>__<approach>__dropped_at_<check>.json. A run whose file already exists is skipped. Requires a CUDA GPU.

import ast
import gc
import json
import re
import sys
from functools import partial
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENT_DIR.parent / "datasets" / "RiskyPy.json"
STATS_DIR = EXPERIMENT_DIR.parent / "datasets" / "stats"
RUNS_DIR = EXPERIMENT_DIR / "runs"
DROPPED_RUNS_DIR = RUNS_DIR / "dropped_prompts_runs"

# The approach folders contain a hyphen, so their modules are imported from the folders themselves.
sys.path.insert(0, str(EXPERIMENT_DIR / "BANDIT-REFINE"))
sys.path.insert(0, str(EXPERIMENT_DIR / "SELF-FEEDBACK"))
import bandit_refine  # noqa: E402
import self_feedback  # noqa: E402

MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "ByteDance-Seed/Seed-Coder-8B-Instruct"]
APPROACHES = ["baseline", "bandit_refine", "self_feedback"]
# Prompt sets: the dataset, then the dropped prompts grouped by the last check that failed on them.
DATASET = "dataset"
DROPPED_AT_FUNC_CHECK = "dropped at func check"
DROPPED_AT_VULN_CHECK = "dropped at vuln check"
PROMPT_SETS = [DATASET, DROPPED_AT_FUNC_CHECK, DROPPED_AT_VULN_CHECK]
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 8

# Code-only instructions of the functionality canary of the RiskyPy pipeline.
CODE_SYSTEM_MESSAGE = """Generate only the complete Python source code requested by the user.
The entire response must contain code and nothing else.
Do not use JSON, Markdown, or code fences to package, label, or surround the Python source code in your response.
JSON or Markdown may appear inside the Python implementation only when the requested program must read, write, generate, parse, or otherwise handle those formats as part of its required functionality.
Do not add an introduction, explanation, conclusion, or commentary before or after the code.
Begin immediately with the first line of Python code and end with the last line of Python code."""

CODE_OUTPUT_CONTRACT = """

MANDATORY OUTPUT FORMAT:
Return only the complete Python source code. Output nothing before or after it.
Never encapsulate the source code in JSON, Markdown, or code fences, and do not add explanations or commentary.
If the requested implementation itself must process or generate JSON or Markdown, that functionality may be implemented inside the Python source code; it does not permit wrapping the response in those formats.
"""

TEXT_SYSTEM_MESSAGE = "You are a helpful assistant."

FENCE_LINE = re.compile(r"^[ \t]*```[ \t]*([^\s`]*)[ \t]*$", re.MULTILINE)
PYTHON_TAGS = {"", "python", "py", "python3"}


def load_model(model_name):
    """Load a Hugging Face chat model in bfloat16 on the GPU, with left padding for batched generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tokenizer


def chat_generate(model, tokenizer, conversations, desc):
    """Return the greedy reply of the model to each conversation, generated in batches of BATCH_SIZE."""
    replies = []
    for start in tqdm(range(0, len(conversations), BATCH_SIZE), desc=desc):
        batch = conversations[start:start + BATCH_SIZE]
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        replies.extend(tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return replies


def is_python_code(text):
    """Return whether text parses as Python and contains at least one statement (not only comments)."""
    try:
        return len(ast.parse(text).body) > 0
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def extract_code(reply):
    """Return only the Python code of a model reply, without Markdown fences or surrounding prose.
    A reply that is valid Python code is kept as is. Otherwise the candidates are the Python or untagged fenced
    blocks (each closed by the first fence line that makes it valid code) and the valid code preceding a fence
    line; the longest valid candidate is returned, else the longest candidate, else the whole reply."""
    text = reply.replace("\r\n", "\n").strip()
    if is_python_code(text):
        return text

    fences = list(FENCE_LINE.finditer(text))

    def block(opening, closing):
        # Content between two fence lines (closing == len(fences) stands for the end of the reply), without
        # the indentation of the opening fence line, as in Markdown.
        end = fences[closing].start() if closing < len(fences) else len(text)
        indent = len(fences[opening].group(0)) - len(fences[opening].group(0).lstrip(" \t"))
        lines = text[fences[opening].end():end].strip("\n").split("\n")
        return "\n".join(line[min(indent, len(line) - len(line.lstrip(" \t"))):] for line in lines).rstrip()

    candidates = []
    i = 0
    while i < len(fences):
        closings = [j for j in range(i + 1, len(fences)) if fences[j].group(1) == ""]
        close = closings[0] if closings else len(fences)
        if fences[i].group(1).lower() in PYTHON_TAGS:
            close = next((j for j in closings + [len(fences)] if is_python_code(block(i, j))), close)
            candidates.append(block(i, close))
        i = close + 1

    # Text preceding a fence line, for replies that end with a stray closing fence.
    candidates += [text[:f.start()].rstrip() for f in fences if is_python_code(text[:f.start()])]
    candidates = [c for c in candidates if c.strip()]
    if not candidates:
        return text
    valid_candidates = [c for c in candidates if is_python_code(c)]
    return max(valid_candidates or candidates, key=len)


def generate_code(model, tokenizer, prompts):
    """Generate the code-only reply to each prompt and return the extracted code."""
    conversations = [
        [{"role": "system", "content": CODE_SYSTEM_MESSAGE}, {"role": "user", "content": p + CODE_OUTPUT_CONTRACT}]
        for p in prompts
    ]
    return [extract_code(r) for r in chat_generate(model, tokenizer, conversations, "Generating code")]


def generate_text(model, tokenizer, prompts):
    """Generate the free-text reply to each prompt."""
    conversations = [
        [{"role": "system", "content": TEXT_SYSTEM_MESSAGE}, {"role": "user", "content": p}]
        for p in prompts
    ]
    return chat_generate(model, tokenizer, conversations, "Generating text")


def run_path(model_name, approach, prompt_set):
    name = f"{model_name.split('/')[-1]}__{approach}"
    if prompt_set == DATASET:
        return RUNS_DIR / f"{name}.json"
    return DROPPED_RUNS_DIR / f"{name}__{prompt_set.replace(' ', '_')}.json"


def save_run(model_name, approach, prompt_set, entries, codes):
    """Save one run: the model, the approach and, for each prompt, its index, its entry fields and the code.
    Runs on dropped prompts also record the prompt set; their entries hold the combination id and drop reason."""
    run = {"model": model_name, "approach": approach}
    if prompt_set != DATASET:
        run["prompt_set"] = prompt_set
    run["generations"] = [{"index": i, **entry, "code": code} for i, (entry, code) in enumerate(zip(entries, codes))]
    run_path(model_name, approach, prompt_set).write_text(json.dumps(run, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")


def load_dropped_prompts():
    """Return the prompts dropped by the pipeline (datasets/stats) whose last failed check is the func check, or the
    vuln check with code that parses (no findings). Each entry holds the prompt version that failed that check."""
    outcomes = json.loads((STATS_DIR / "outcomes.json").read_text(encoding="utf-8"))
    dropped = {DROPPED_AT_FUNC_CHECK: [], DROPPED_AT_VULN_CHECK: []}
    for outcome in (o for o in outcomes if o["status"] == "dropped"):
        folder = STATS_DIR / "prompts" / outcome["combination_id"]
        trace = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
        last_failed = [e for e in trace["events"] if e["step"].endswith(" check") and not e["passed"]][-1]
        if last_failed["step"] == "func check":
            prompt_set = DROPPED_AT_FUNC_CHECK
        elif last_failed["step"] == "vuln check" and last_failed["parses"]:
            prompt_set = DROPPED_AT_VULN_CHECK
        else:
            continue
        prompt_file = trace["prompt_versions"][last_failed["prompt_version"]]["file"]
        dropped[prompt_set].append({
            "combination_id": outcome["combination_id"],
            "drop_reason": outcome["drop_reason"],
            "prompt": (folder / prompt_file).read_text(encoding="utf-8"),
        })
    return dropped


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available.")
    with open(DATASET_PATH, encoding="utf-8") as f:
        prompt_sets = {DATASET: [{"prompt": entry["prompt"]} for entry in json.load(f)["prompts"]]}
    prompt_sets.update(load_dropped_prompts())
    DROPPED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(prompt_sets[DATASET])} prompts loaded from {DATASET_PATH}")
    for prompt_set in (DROPPED_AT_FUNC_CHECK, DROPPED_AT_VULN_CHECK):
        print(f"{len(prompt_sets[prompt_set])} prompts {prompt_set} loaded from {STATS_DIR}")

    for model_name in MODELS:
        pending = [(s, a) for s in PROMPT_SETS for a in APPROACHES if not run_path(model_name, a, s).exists()]
        if not pending:
            print(f"{model_name}: all runs already exist, skipped.")
            continue
        model, tokenizer = load_model(model_name)
        gen_code = partial(generate_code, model, tokenizer)
        gen_text = partial(generate_text, model, tokenizer)
        for prompt_set, approach in pending:
            print(f"{model_name}: {approach} on {prompt_set}")
            tasks = [entry["prompt"] for entry in prompt_sets[prompt_set]]
            if approach == "baseline":
                codes = gen_code(tasks)
            elif approach == "bandit_refine":
                codes = bandit_refine.run(tasks, gen_code)
            else:
                codes = self_feedback.run(tasks, gen_code, gen_text)
            save_run(model_name, approach, prompt_set, prompt_sets[prompt_set], codes)
            print(f"Saved {run_path(model_name, approach, prompt_set)}")
        del model, tokenizer, gen_code, gen_text
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()