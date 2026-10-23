# Computes the results of the manual annotation from the three annotations_<name>.json files next to this script.
# Prompt quality: mean rating and its standard deviation, mean absolute difference per pair of annotators, ICC(2,1),
# Krippendorff's alpha (interval). Functionality: Krippendorff's alpha (nominal) and pairwise Cohen's kappa between the
# annotators, and the agreement of the LLM judge (results/judgments.json) with the majority label and each annotator.
# A skipped generation (null decision) is left out of the comparisons with its annotator; majority ties are left out.

import json
import statistics
from itertools import combinations
from pathlib import Path

ANNOTATION_DIR = Path(__file__).resolve().parent
SELECTION_PATH = ANNOTATION_DIR / "selection.json"
JUDGMENTS_PATH = ANNOTATION_DIR.parent / "results" / "judgments.json"
ANNOTATOR_COUNT = 3


def load_annotations(selection):
    """Return (annotator, grades, decisions) for each of the three annotation files, in file name order.
    grades and decisions map each sample id to its answer; every prompt must be graded, a generation may be skipped."""
    paths = sorted(ANNOTATION_DIR.glob("annotations_*.json"))
    if len(paths) != ANNOTATOR_COUNT:
        raise SystemExit(f"Expected {ANNOTATOR_COUNT} annotations_*.json files in {ANNOTATION_DIR}, "
                         f"found {len(paths)}.")
    annotations = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        grades = {e["id"]: e["grade"] for e in data["prompt_quality"]}
        decisions = {e["id"]: e["is_functional"] for e in data["functionality"]}
        for part, answers in (("prompt_quality", grades), ("functionality", decisions)):
            if list(answers) != [s["id"] for s in selection[part]]:
                raise SystemExit(f"{path.name} does not contain the samples of selection.json.")
            missing = [sample_id for sample_id, answer in answers.items() if answer is None]
            if missing and part == "prompt_quality":
                raise SystemExit(f"{path.name} has no answer for {', '.join(missing)}.")
            if missing:
                print(f"Note: {data['annotator']} skipped {', '.join(missing)}; "
                      f"left out of the comparisons involving {data['annotator']}.")
        annotations.append((data["annotator"], grades, decisions))
    return annotations


def judge_decisions(selection):
    """Return the judge decision of each annotated generation, keyed by sample id, read from judgments.json.
    A judge failure during the evaluation (null decision) counts as not functional, as in results.json."""
    if not JUDGMENTS_PATH.exists():
        raise SystemExit(f"{JUDGMENTS_PATH} not found. Run evaluate_runs.py first.")
    judgments = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))["judgments"]
    by_generation = {(j["run_file"], j["index"]): j for j in judgments}
    decisions, failed = {}, []
    for s in selection["functionality"]:
        # The saved judgment must be of the annotated generation: same run, index, prompt and code.
        j = by_generation.get((s["run_file"], s["dataset_index"]))
        if j is None or (j["prompt"], j["code"]) != (s["prompt"], s["code"]):
            raise SystemExit(f"{JUDGMENTS_PATH.name} does not contain the generation {s['id']} of selection.json.")
        if j["is_functional"] is None:
            failed.append(s["id"])
        decisions[s["id"]] = j["is_functional"] is True
    if failed:
        print(f"Note: the judge failed on {', '.join(failed)} during the evaluation; counted as not functional.")
    return decisions


def icc_2_1(ratings):
    """ICC(2,1) of Shrout and Fleiss (two-way random effects, absolute agreement, single rater).
    ratings has one row per rated item and one column per rater. None when all ratings are identical."""
    n, k = len(ratings), len(ratings[0])
    grand_mean = sum(map(sum, ratings)) / (n * k)
    row_means = [sum(row) / k for row in ratings]
    column_means = [sum(row[j] for row in ratings) / n for j in range(k)]
    ss_rows = k * sum((m - grand_mean) ** 2 for m in row_means)
    ss_columns = n * sum((m - grand_mean) ** 2 for m in column_means)
    ss_total = sum((x - grand_mean) ** 2 for row in ratings for x in row)
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = (ss_total - ss_rows - ss_columns) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return (ms_rows - ms_error) / denominator if denominator else None


def answered_pairs(a, b):
    """Pairs of answers of two equally long decision lists, without the positions where either answer is None."""
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def cohen_kappa(a, b):
    """Cohen's kappa between two equally long lists of binary decisions, on the positions both answered.
    None when it is undefined, that is when both lists contain only one and the same decision."""
    pairs = answered_pairs(a, b)
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    positive_a, positive_b = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    expected = positive_a * positive_b + (1 - positive_a) * (1 - positive_b)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def krippendorff_alpha(items, distance):
    """Krippendorff's alpha; items has one list per rated item with one value (or None if missing) per rater.
    distance(x, y) is the squared difference of two values (nominal or interval below). None when undefined.
    Items with fewer than two values cannot be paired and are left out."""
    items = [[v for v in item if v is not None] for item in items]
    items = [item for item in items if len(item) >= 2]
    values = [v for item in items for v in item]
    n = len(values)
    # Observed disagreement: within-item pairs, each item weighted by 1 / (its value count - 1).
    observed = sum(sum(distance(x, y) for i, x in enumerate(item) for j, y in enumerate(item) if i != j)
                   / (len(item) - 1) for item in items) / n
    # Expected disagreement: all pairs of values, regardless of the item.
    expected = sum(distance(x, y) for i, x in enumerate(values) for j, y in enumerate(values) if i != j) / (n * (n - 1))
    return 1 - observed / expected if expected else None


def nominal(x, y):
    return 0.0 if x == y else 1.0


def interval(x, y):
    return float((x - y) ** 2)


def formatted(value):
    return "undefined" if value is None else f"{value:.3f}"


def mean_or_none(values):
    """Mean of the values, or None when one of them is undefined (None)."""
    return None if None in values else statistics.mean(values)


def agreement(a, b):
    """Raw agreement (count and percentage) and Cohen's kappa of two decision lists on the positions both answered,
    as printed text."""
    pairs = answered_pairs(a, b)
    agreeing = sum(x == y for x, y in pairs)
    return (f"raw agreement {agreeing}/{len(pairs)} ({100 * agreeing / len(pairs):.1f}%), "
            f"Cohen's kappa {formatted(cohen_kappa(a, b))}")


def main():
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    annotations = load_annotations(selection)
    names = [name for name, _, _ in annotations]
    pairs = list(combinations(range(len(names)), 2))

    # One row per prompt, one column per annotator.
    prompt_ids = [s["id"] for s in selection["prompt_quality"]]
    ratings = [[grades[i] for _, grades, _ in annotations] for i in prompt_ids]
    print(f"Prompt quality ({len(prompt_ids)} prompts, annotators {', '.join(names[:-1])} and {names[-1]})")
    all_ratings = [x for row in ratings for x in row]
    print(f"  Mean rating: {statistics.mean(all_ratings):.2f} "
          f"(standard deviation over all ratings {statistics.stdev(all_ratings):.2f})")
    for k, name in enumerate(names):
        print(f"  Mean rating of {name}: {statistics.mean(row[k] for row in ratings):.2f}")
    # Quality score of a prompt: the mean of its ratings. Sample standard deviation over the prompts.
    prompt_scores = [statistics.mean(row) for row in ratings]
    print(f"  Standard deviation of the per-prompt mean ratings: {statistics.stdev(prompt_scores):.2f}")
    for a, b in pairs:
        differences = [abs(row[a] - row[b]) for row in ratings]
        print(f"  Mean absolute difference {names[a]} vs {names[b]}: {statistics.mean(differences):.2f} "
              f"(standard deviation {statistics.stdev(differences):.2f})")
    print(f"  ICC(2,1): {formatted(icc_2_1(ratings))}")
    print(f"  Krippendorff's alpha (interval): {formatted(krippendorff_alpha(ratings, interval))}")

    generation_ids = [s["id"] for s in selection["functionality"]]
    judge_by_id = judge_decisions(selection)
    judge = [judge_by_id[i] for i in generation_ids]
    columns = [[decisions[i] for i in generation_ids] for _, _, decisions in annotations]
    items = [list(item) for item in zip(*columns)]
    # Majority label: the decision given by more annotators than the opposite one; None on a tie.
    majority = []
    for item in items:
        positive, negative = item.count(True), item.count(False)
        majority.append(None if positive == negative else positive > negative)
    unanimous = [k for k, item in enumerate(items) if len(set(item)) == 1]
    print(f"\nFunctionality ({len(generation_ids)} generations)")
    counts = ", ".join(f"{name} {column.count(True)}/{len(column) - column.count(None)}"
                       for name, column in zip(names, columns))
    print(f"  Judged functional: LLM judge {sum(judge)}/{len(judge)}, {counts}")
    print(f"  Krippendorff's alpha (nominal) between the annotators: {formatted(krippendorff_alpha(items, nominal))}")
    kappas = []
    for a, b in pairs:
        kappas.append(cohen_kappa(columns[a], columns[b]))
        n = len(answered_pairs(columns[a], columns[b]))
        print(f"  Cohen's kappa {names[a]} vs {names[b]}: {formatted(kappas[-1])} (n={n})")
    print(f"  Mean pairwise Cohen's kappa between the annotators: {formatted(mean_or_none(kappas))}")
    print(f"  LLM judge vs majority of the annotators (ties left out: {majority.count(None)}): "
          f"{agreement(judge, majority)}")
    kappas = []
    for name, column in zip(names, columns):
        kappas.append(cohen_kappa(judge, column))
        print(f"  LLM judge vs {name}: {agreement(judge, column)}")
    print(f"  Mean Cohen's kappa of the LLM judge vs each annotator: {formatted(mean_or_none(kappas))}")
    agreeing = sum(judge[k] == majority[k] for k in unanimous)
    share = f" ({100 * agreeing / len(unanimous):.1f}%)" if unanimous else ""
    print(f"  Unanimous generations: {len(unanimous)}/{len(generation_ids)}, "
          f"LLM judge agrees on {agreeing}/{len(unanimous)}{share}")


if __name__ == "__main__":
    main()