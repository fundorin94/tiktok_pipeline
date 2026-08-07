from importlib import import_module

STAGE_MODULES = {
    "story": "agents.story_research",
    "script": "agents.script_writer",
    "archive": "agents.archive_finder",
    "voiceover": "agents.voiceover",
    "video": "agents.video_assembly",
    "metadata": "agents.metadata",
    "publish": "agents.publisher",
}


def run_stage(case_id: str, stage: str, db) -> None:
    if stage not in STAGE_MODULES:
        raise ValueError(f"Unknown stage: {stage} (known stages: {list(STAGE_MODULES)})")

    module = import_module(STAGE_MODULES[stage])
    db.log_stage(case_id, stage, "running")
    try:
        module.run(case_id, db)
    except Exception as exc:
        db.log_stage(case_id, stage, "error", str(exc))
        raise
    else:
        db.log_stage(case_id, stage, "done")
