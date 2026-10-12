"""
================================================================================
FULL EXPERIMENT SWEEP - CONCURRENT DRIVER
================================================================================
Runs the entire grid: every dataset x every identity condition x every sampled
question, as independent SimulationRunner jobs. Each job is internally
sequential (its own 15-stage pipeline), but the jobs themselves have zero
dependency on each other, so they run concurrently via a thread pool - this is
what keeps the vLLM servers saturated with batched work instead of processing
one call at a time.

Resumable: a job whose output file already exists is skipped, so an
interrupted sweep can just be re-run and will only redo what's missing.

Usage:
  python3 run_full_experiment.py --provider vllm --concurrency 40
  python3 run_full_experiment.py --provider mock --concurrency 8 --questions-per-dataset 2  # smoke test
================================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from dataset_loaders import get_dataset
from identities import VIRTUAL_PERSONA_COMBOS, FULL_PERSONA_COMBO
from run_experiment import (
    SimulationRunner,
    compute_output_label,
    LOG_OUTPUT_ROOT,
    NUM_DEBATE_ROUNDS,
    USE_HUGGINGFACE_ONLINE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("full_sweep")

# The two identity conditions of the study: (identity_mode, virtual_combo_or_None).
#   anonymous           - agents see only names such as Agent_06
#   real_full_identity  - agents also see each model's name, version and size
ALL_IDENTITY_CONDITIONS: List[Tuple[str, Optional[str]]] = [
    ("anonymous", None),
    ("real_full_identity", None),
]

ALL_DATASETS = ["moralchoice", "moralmachine", "mmlu"]

_progress_lock = Lock()
_progress = {"completed": 0, "skipped": 0, "failed": 0, "total": 0}


def build_jobs(
    datasets: List[str],
    questions_per_dataset: int,
    conditions: List[Tuple[str, Optional[str]]],
    use_huggingface: bool
) -> List[Dict[str, Any]]:
    jobs = []
    for ds in datasets:
        items = get_dataset(ds, max_samples=questions_per_dataset, use_huggingface=use_huggingface)
        logger.info(f"Dataset '{ds}': {len(items)} questions selected")
        for item in items:
            for identity_mode, combo in conditions:
                jobs.append({
                    "question_item": item,
                    "identity_mode": identity_mode,
                    "virtual_combo": combo or FULL_PERSONA_COMBO,
                })
    return jobs


def _condition_label(res: Dict[str, Any]) -> str:
    """Human-readable condition label for log lines - disambiguates the 7 virtual_persona combos."""
    mode = res.get("identity_mode")
    combo = res.get("virtual_combo")
    return f"{mode} ({combo})" if combo else str(mode)


def expected_output_path(dataset_dir: str, problem_id: str, identity_mode: str, virtual_combo: Optional[str]) -> str:
    label = compute_output_label(identity_mode, virtual_combo=virtual_combo)
    return os.path.join(dataset_dir, f"{problem_id}_{label}.json")


def _job_already_completed(out_path: str) -> bool:
    """
    A job's .json file is written the moment its ConversationLogger is
    constructed (near the very start of a run, to support live tailing), not
    only on success - so file existence alone does NOT mean the job finished.
    Confirmed on a real run: a job that crashed mid-baseline-stage still left a
    .json on disk with just the problem header and an empty summary_analytics.
    finalize_simulation() is the only thing that ever populates
    summary_analytics, so a non-empty dict there is the real completion marker.
    A missing/corrupt/unreadable file is treated as not completed (re-run it).
    """
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return bool(doc.get("summary_analytics"))
    except (json.JSONDecodeError, OSError):
        return False


def run_one_job(job: Dict[str, Any], provider: str, model: Optional[str], num_debate_rounds: int) -> Dict[str, Any]:
    item = job["question_item"]
    identity_mode = job["identity_mode"]
    virtual_combo = job["virtual_combo"]
    dataset_name = item.get("dataset", "custom").split("/")[-1]
    problem_id = item["id"].replace("/", "_")
    dataset_dir = os.path.join(LOG_OUTPUT_ROOT, dataset_name)

    out_path = expected_output_path(dataset_dir, problem_id, identity_mode, virtual_combo)
    if _job_already_completed(out_path):
        return {"status": "skipped", "path": out_path}

    try:
        runner = SimulationRunner(
            question_item=item,
            identity_mode=identity_mode,
            virtual_combo=virtual_combo,
            num_debate_rounds=num_debate_rounds,
            provider=provider,
            model=model,
            output_root=LOG_OUTPUT_ROOT
        )
        result = runner.run()
        return {"status": "completed", "path": result["json_path"]}
    except Exception as e:
        return {
            "status": "failed",
            "problem_id": problem_id,
            "identity_mode": identity_mode,
            "virtual_combo": virtual_combo if identity_mode == "virtual_persona" else None,
            "error": f"{type(e).__name__}: {e}"
        }


def main():
    parser = argparse.ArgumentParser(description="Run the full dataset x identity-condition experiment sweep")
    parser.add_argument("--datasets", type=str, default=",".join(ALL_DATASETS), help="Comma-separated dataset names")
    parser.add_argument("--questions-per-dataset", type=int, default=50, help="Stratified questions to sample per dataset")
    parser.add_argument("--concurrency", type=int, default=40, help="Number of simulations to run concurrently")
    parser.add_argument("--provider", type=str, default="vllm", help="Backend provider (vllm, mock, ollama, etc.)")
    parser.add_argument("--model", type=str, default=None, help="Optional: force all agents to a single model name")
    parser.add_argument("--rounds", type=int, default=NUM_DEBATE_ROUNDS, help="Debate rounds per simulation (default 11)")
    parser.add_argument("--use-huggingface", action="store_true", default=USE_HUGGINGFACE_ONLINE, help="Fetch datasets live from HuggingFace instead of the built-in fallback")
    parser.add_argument("--identity-conditions", type=str, default=None,
                         help="Comma-separated identity modes to restrict this run to (e.g. 'anonymous'). "
                              "Defaults to all conditions if omitted.")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.identity_conditions:
        wanted = {c.strip() for c in args.identity_conditions.split(",") if c.strip()}
        conditions = [c for c in ALL_IDENTITY_CONDITIONS if c[0] in wanted]
    else:
        conditions = ALL_IDENTITY_CONDITIONS

    jobs = build_jobs(datasets, args.questions_per_dataset, conditions, args.use_huggingface)
    _progress["total"] = len(jobs)

    logger.info("=" * 70)
    logger.info(f"FULL SWEEP: {len(jobs)} total jobs "
                f"({len(datasets)} datasets, {len(conditions)} identity condition(s))")
    logger.info(f"Provider: {args.provider} | Concurrency: {args.concurrency} | Debate rounds: {args.rounds}")
    logger.info("=" * 70)

    start_time = time.time()
    failures = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one_job, job, args.provider, args.model, args.rounds): job
            for job in jobs
        }
        for future in as_completed(futures):
            res = future.result()
            with _progress_lock:
                if res["status"] == "skipped":
                    _progress["skipped"] += 1
                elif res["status"] == "completed":
                    _progress["completed"] += 1
                else:
                    _progress["failed"] += 1
                    failures.append(res)
                    logger.error(f"[FAILED] {res.get('problem_id')} / {_condition_label(res)}: {res.get('error')}")

                done = _progress["completed"] + _progress["skipped"] + _progress["failed"]
                if done % 10 == 0 or done == _progress["total"]:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Progress: {done}/{_progress['total']} "
                        f"(completed={_progress['completed']}, skipped={_progress['skipped']}, failed={_progress['failed']}) "
                        f"| elapsed {elapsed/60:.1f} min"
                    )

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"SWEEP FINISHED in {elapsed/60:.1f} min")
    logger.info(f"Completed: {_progress['completed']} | Skipped (already done): {_progress['skipped']} | Failed: {_progress['failed']}")
    if failures:
        logger.info("Failed jobs (re-run the sweep to retry only these - completed ones are skipped):")
        for f in failures:
            logger.info(f"  - {f.get('problem_id')} / {_condition_label(f)}: {f.get('error')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
