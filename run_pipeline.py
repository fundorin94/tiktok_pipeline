import argparse

from config import CASES_DIR, DB_PATH
from orchestrator.cost import estimate_cost
from orchestrator.db import Database
from orchestrator.pipeline import STAGE_MODULES, run_stage


def main():
    parser = argparse.ArgumentParser(description="Run the true-crime content pipeline for one case.")
    parser.add_argument("--case-id", required=True, help="Unique slug for this case, e.g. test1")
    parser.add_argument("--topic", default=None, help="Optional topic/theme hint for story research")
    parser.add_argument(
        "--stage",
        default="story",
        choices=list(STAGE_MODULES.keys()) + ["all"],
        help="Which pipeline stage to run",
    )
    args = parser.parse_args()

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DB_PATH)
    db.create_case(args.case_id, args.topic)

    stages = list(STAGE_MODULES.keys()) if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"[{args.case_id}] running stage: {stage}")
        run_stage(args.case_id, stage, db)
        print(f"[{args.case_id}] stage done: {stage}")

    cost = estimate_cost(db.get_usage(args.case_id))
    print(f"[{args.case_id}] estimated API cost so far (this case): ${cost:.4f}")
    total_cost = estimate_cost(db.get_usage())
    print(f"estimated API cost so far (all cases, this local DB): ${total_cost:.4f}")


if __name__ == "__main__":
    main()
