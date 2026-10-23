# RiskyPy

RiskyPy generates Python coding tasks for evaluating security and functionality together. It combines 12 risky programming domains into 298 combinations of one, two or three domains, and writes one prompt per combination. Each prompt must pass three checks, in order:

- **Quality check:** an LLM judge verifies that the prompt is realistic, consistent and does not prescribe a security-relevant mechanism.
- **Func check:** a small model (Ministral 3B) writes code for the prompt, and an LLM judge verifies that the code is functional.
- **Vuln check:** a stronger model (GPT-4o-mini) writes code for the prompt, and Bandit or Semgrep must report at least one finding on it.

A failed check triggers a rewrite of the prompt (adjustment, simplification or risky reframing), and the rewritten prompt goes through every check again. A combination is dropped when a rewrite limit is reached or when the vuln check code does not parse. Similar prompts are then removed. The repository includes the resulting dataset of 101 prompts, an ablation dataset where every rewrite is a plain rephrasing, and experiments with two code-generation models.

## Folders

| Folder | Contents |
| --- | --- |
| `pipeline/` | Dataset generation script, rephrasing ablation script, domain definitions, and dependencies. |
| `datasets/` | `RiskyPy.json` (101 prompts) and `RephraseAblationDataset.json` (61 prompts), with their canary code and static-analysis findings. |
| `datasets/stats/` | Run statistics, outcome of each combination, log, and every prompt version with its trace, for `RiskyPy.json`. |
| `datasets/statsAblation/` | The same files for `RephraseAblationDataset.json`. |
| `experiment/` | Model generation and evaluation scripts and dependencies. |
| `experiment/BANDIT-REFINE/` | One code-repair pass using Bandit findings. |
| `experiment/SELF-FEEDBACK/` | Security review and code rewriting by the generating model. |
| `experiment/runs/` | Saved code generations on the dataset prompts, for each model and approach. |
| `experiment/runs/dropped_prompts_runs/` | Saved code generations on the prompts the pipeline dropped at the func check (176 prompts) or at the vuln check (4 prompts). |
| `experiment/results/` | Evaluations, statistics with 95% confidence intervals, functionality judgments, CWE summaries, dataset versus dropped prompts comparisons, and charts. |
| `experiment/annotation/` | Manual annotation app, sample selection, annotations of three annotators, and agreement analysis. |
| `colabrunner/` | Google Colab notebook for running the model-generation experiments. |

## Run

Commands below run from the repository root in a Python environment. API steps require network access and the corresponding API keys. Semgrep downloads its rules from the registry.

### Generate the dataset

```bash
python -m pip install -r pipeline/requirements.txt
export OPENAI_API_KEY="<key>"
export OPENROUTER_API_KEY="<key>"
python pipeline/RiskyPyPipeline.py
```

This writes `datasets/RiskyPy.json`, replacing any existing dataset, and deletes and recreates `datasets/stats/`.

The rephrasing ablation runs the same way with `python pipeline/RephraseAblationPipeline.py`. It writes `datasets/RephraseAblationDataset.json` and `datasets/statsAblation/`.

Both scripts read three optional environment variables:

- `RISKYPY_OUTPUT_PATH`: writes the dataset to another path. The statistics folder is created next to it.
- `RISKYPY_MAX_COMBINATIONS`: processes only the first N combinations.
- `RISKYPY_MAX_WORKERS`: number of combinations processed in parallel (6 by default).

### Generate and evaluate model outputs

```bash
python -m pip install -r experiment/requirements.txt
python experiment/run_experiment.py
export OPENAI_API_KEY="<key>"
python experiment/evaluate_runs.py
```

Generation requires a CUDA GPU. It runs Qwen3-4B and Seed-Coder-8B with the baseline, BANDIT-REFINE, and SELF-FEEDBACK approaches, on the dataset prompts and on the dropped prompts. The dropped prompts are read from `datasets/stats/`. Existing run files are skipped. Move the included files out of `experiment/runs/` and `experiment/runs/dropped_prompts_runs/` before generating fresh runs.

Evaluation can run directly on the included runs without a GPU. It uses Bandit, Semgrep, and a GPT-5.1 functionality judge, and rewrites the files in `experiment/results/`. Evaluations are saved to `experiment/results/evaluations.json` during the run and reused on the next run when the prompt and code are unchanged, so an interrupted evaluation continues where it stopped. With the included `evaluations.json`, the script only recomputes the statistics and charts. Delete this file to evaluate every generation again.

`experiment/clean.py` removes the last 90 saved entries of `evaluations.json` so that they are evaluated again. It runs from the `experiment/` folder and does nothing once evaluations of the prompts dropped at the vuln check are saved.

The Colab notebook provides an alternative for model generation. It takes a zip of the repository, runs `run_experiment.py`, and saves the repository with its runs as a zip in Google Drive and on the local machine. Evaluation runs separately.

### Manual annotation

```bash
python -m pip install -r experiment/annotation/requirements.txt
python -m streamlit run experiment/annotation/annotate.py --server.address localhost
python experiment/annotation/analyze_annotations.py
```

The app uses the included `selection.json`: 30 prompts graded for quality from 0 to 10, and 30 generations judged functional or not. Each annotator's answers are saved to `annotations_<name>.json` in the same folder. `experiment/annotation/README.md` gives the instructions for annotators.

`analyze_annotations.py` needs exactly three annotation files and `experiment/results/judgments.json`. It prints the agreement between the annotators and between the annotators and the LLM judge. `select_annotation_samples.py` creates a new selection when needed.
