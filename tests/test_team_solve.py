"""Unit tests for Memdoor-style decentralized multi-agent team-solve workflow."""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.team_solve import (
    TeamSolveWorkflow,
    TaskLedger,
    AgentRole,
    WorkflowVerdict,
    ModelConfig,
)


@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "team_solve_ws"
    ws.mkdir()
    return ws


def test_task_ledger_initialization_and_patching(temp_workspace):
    ledger = TaskLedger(temp_workspace)
    ledger.initialize_notes("test-partition-001")

    assert ledger.notes_path.exists()
    text = ledger.notes_path.read_text(encoding="utf-8")
    assert "<!-- proposal-a -->" in text
    assert "<!-- proposal-b -->" in text
    assert "<!-- solver-decision -->" in text

    # Test patch insertion before marker
    patch = "### Proposal A Content: Slope protection rules"
    ok = ledger.insert_before_marker("<!-- proposal-a -->", patch)
    assert ok is True

    updated_text = ledger.notes_path.read_text(encoding="utf-8")
    assert patch in updated_text
    assert "<!-- proposal-a -->" in updated_text
    # Verify patch appears BEFORE marker
    assert updated_text.index(patch) < updated_text.index("<!-- proposal-a -->")


def test_team_solve_workflow_end_to_end(temp_workspace):
    workflow = TeamSolveWorkflow(
        workspace_dir=temp_workspace,
        use_mock_inference_if_offline=True,
    )

    task_spec = (
        "Solve European Soil-Water Nexus Pareto Reallocation: "
        "Preserve agricultural yield while mitigating soil loss."
    )

    verdict = workflow.run_workflow(task_spec)

    assert isinstance(verdict, WorkflowVerdict)
    assert verdict.success is True
    assert verdict.exit_code == 0
    assert "init_ledger" in verdict.step_history
    assert "plan" in verdict.step_history
    assert "ideate_a" in verdict.step_history
    assert "ideate_b" in verdict.step_history
    assert "solve" in verdict.step_history
    assert "test" in verdict.step_history
    assert "review" in verdict.step_history
    assert "verify" in verdict.step_history

    # Verify all artifacts on disk
    assert (temp_workspace / "task.md").exists()
    assert (temp_workspace / "plan.md").exists()
    assert (temp_workspace / "notes.md").exists()
    assert (temp_workspace / "solution.py").exists()
    assert (temp_workspace / "test-status.txt").exists()
    assert (temp_workspace / "test-output.log").exists()
    assert (temp_workspace / "review.md").exists()

    status_content = (temp_workspace / "test-status.txt").read_text(encoding="utf-8").strip()
    assert status_content == "0"
