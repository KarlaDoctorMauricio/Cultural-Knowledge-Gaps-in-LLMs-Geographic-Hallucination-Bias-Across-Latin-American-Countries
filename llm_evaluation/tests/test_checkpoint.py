import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import classify_low_score_responses  # noqa: E402
from fairness_toolkit.checkpoint import judge_done_keys, load_judge_checkpoint  # noqa: E402


def test_judge_checkpoint_resume(tmp_path):
    df = pd.DataFrame(
        {
            "Question": ["Q1", "Q2"],
            "Answer": ["A1", "A2"],
            "response_GPT": ["R1", "R2"],
            "similarity_GPT": [0.2, 0.9],
        }
    )
    calls = {"count": 0}

    def fake_judge(_prompt):
        calls["count"] += 1
        return "CORRECTA"

    classify_low_score_responses(
        df.iloc[:1],
        model_names=("GPT",),
        primary_query_fn=fake_judge,
        backup_query_fn=fake_judge,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
        resume=True,
    )
    assert calls["count"] == 1

    calls["count"] = 0
    classified = classify_low_score_responses(
        df,
        model_names=("GPT",),
        primary_query_fn=fake_judge,
        backup_query_fn=fake_judge,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
        resume=True,
    )

    assert calls["count"] == 1
    assert len(classified) == 2
    assert len(judge_done_keys(load_judge_checkpoint(tmp_path))) == 2
