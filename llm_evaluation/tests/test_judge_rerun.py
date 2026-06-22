import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clients import SYSTEM_PROMPT, query_gpt  # noqa: E402
from evaluator import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, llm_judge_classify  # noqa: E402
from scripts.rerun_judge_stability import (  # noqa: E402
    archive_pre_fix_outputs,
    build_before_after_prompt_fix,
    build_judge_stability_table,
    consolidate_judge_runs,
)


def test_query_gpt_uses_custom_system_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict = {}

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return mock_client

    with patch("openai.OpenAI", side_effect=fake_openai):
        query_gpt("user msg", system_prompt="custom system")

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == "custom system"
    assert messages[1]["content"] == "user msg"


def test_llm_judge_passes_system_separately(monkeypatch):
    seen: dict = {}

    def fake_gpt(user_prompt, *, system_prompt):
        seen["user"] = user_prompt
        seen["system"] = system_prompt
        return "CORRECTA"

    label, _, method = llm_judge_classify(
        question="Q?",
        reference="A",
        response="R",
        score=0.8,
        primary_query_fn=fake_gpt,
        backup_query_fn=lambda *_a, **_k: None,
    )
    assert label == "correcta"
    assert method == "gpt"
    assert "CORRECTA:" in seen["system"]
    assert "Clasifica la respuesta del modelo en UNA sola categoria" not in seen["user"]
    assert "Q?" in seen["user"]


def test_judge_user_template_is_data_only():
    user = JUDGE_USER_TEMPLATE.format(
        question="Q",
        reference="A",
        response="R",
        score=0.5,
    )
    assert "Ejemplos orientativos" not in user
    assert "CORRECTA:" not in user.split("\n")[0]


def test_build_judge_stability_table():
    run1 = pd.DataFrame(
        {
            "row_index": [0, 1],
            "model": ["GPT", "GPT"],
            "pais": ["chile", "chile"],
            "judge_label": ["alucinacion", "parcial"],
        }
    )
    run2 = pd.DataFrame(
        {
            "row_index": [0, 1],
            "model": ["GPT", "GPT"],
            "pais": ["chile", "chile"],
            "judge_label": ["alucinacion", "correcta"],
        }
    )
    run3 = pd.DataFrame(
        {
            "row_index": [0, 1],
            "model": ["GPT", "GPT"],
            "pais": ["chile", "chile"],
            "judge_label": ["alucinacion", "parcial"],
        }
    )
    stability, pct = build_judge_stability_table([run1, run2, run3])
    assert pct == pytest.approx(50.0)
    assert bool(stability.iloc[0]["stable"]) is True
    assert bool(stability.iloc[1]["stable"]) is False
    assert stability.iloc[1]["final_label"] == "parcial"


def test_consolidate_judge_runs_uses_mode():
    runs = [
        pd.DataFrame(
            {
                "row_index": [0],
                "model": ["GPT"],
                "pais": ["chile"],
                "judge_label": ["alucinacion"],
                "Question": ["Q"],
            }
        ),
        pd.DataFrame(
            {
                "row_index": [0],
                "model": ["GPT"],
                "pais": ["chile"],
                "judge_label": ["parcial"],
            }
        ),
        pd.DataFrame(
            {
                "row_index": [0],
                "model": ["GPT"],
                "pais": ["chile"],
                "judge_label": ["parcial"],
            }
        ),
    ]
    final = consolidate_judge_runs(runs)
    assert final.iloc[0]["judge_label"] == "parcial"


def test_archive_moves_files(tmp_path):
    for name in ("response_quality_breakdown.csv", "low_score_classifications.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    moved = archive_pre_fix_outputs(tmp_path)
    assert len(moved) == 2
    assert not (tmp_path / "response_quality_breakdown.csv").exists()
    assert (tmp_path / "archive_pre_fix" / "response_quality_breakdown.csv").exists()


def test_before_after_prompt_fix(tmp_path):
    archive = tmp_path / "archive_pre_fix"
    archive.mkdir()
    pd.DataFrame(
        {
            "model": ["GPT"],
            "pct_alucinacion": [28.9],
            "pct_abstencion": [5.0],
        }
    ).to_csv(archive / "response_quality_breakdown.csv", index=False)
    new = pd.DataFrame(
        {
            "model": ["GPT"],
            "pct_alucinacion": [25.0],
            "pct_abstencion": [8.0],
        }
    )
    diff = build_before_after_prompt_fix(archive, new)
    assert diff.iloc[0]["delta_alucinacion"] == pytest.approx(-3.9)
