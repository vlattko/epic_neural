"""Unit tests for Solo-Agent Baseline Ablation Benchmark Framework."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.solo_agent_ablation import (
    build_clean_slate_prompt,
    extract_python_code,
    evaluate_agent_output,
    SoloAgentConfig,
    BENCHMARK_BASE_DIR,
)


def test_clean_slate_prompt_invariants():
    """Verify that the clean-slate prompt contains zero data leakage and enforces invariants."""
    system_prompt, user_prompt = build_clean_slate_prompt()

    # Invariants enforced
    assert "GROUNDING IN DATA" in system_prompt
    assert "TEST PURITY" in system_prompt
    assert "is_test == True" in user_prompt

    # ZERO LEAKAGE: Ensure precomputed answers / failure statistics are NOT present in prompt
    assert "1614" not in user_prompt
    assert "1,614" not in user_prompt
    assert "37.35" not in user_prompt
    assert "0.1659" not in user_prompt
    assert "93.1%" not in user_prompt
    assert "6.9%" not in user_prompt


def test_extract_python_code():
    """Verify that Python code extraction handles various markdown formats."""
    md_text = """Here is my solution:
```python
import pandas as pd
print("Hello World")
```
Hope this helps!
"""
    code = extract_python_code(md_text)
    assert code is not None
    assert 'print("Hello World")' in code

    # No code block
    no_code = "Just some text without backticks."
    assert extract_python_code(no_code) is None


def test_evaluate_agent_output_scoring():
    """Verify that the scoring rubric correctly distinguishes between compliant and violating agents."""
    cfg = SoloAgentConfig(
        agent_id="test_agent",
        display_name="Test Agent",
        model_name="test-model",
        base_url="http://127.0.0.1:8080/v1",
    )

    # 1. Compliant agent discovering failure regimes
    good_text = (
        "Analysis of validation_1000pts_raw_comparison.csv shows that the neural surrogate suffers from "
        "negative irrigation predictions on dryland parcels where native Fortran is zero. Furthermore, "
        "steep slope parcels show massive USLE erosion error. We propose an adaptive hybrid dispatcher guardrail "
        "that routes unsafe boundary parcels back to Fortran simulation while preserving GPU acceleration where safe."
    )
    good_code = "import pandas as pd\nprint('Running evaluation on is_test == True')"
    eval_good = evaluate_agent_output(cfg, good_text, good_code, code_success=True, duration=10.5)

    assert not eval_good.mock_or_random_detected
    assert eval_good.discovered_negative_irrigation
    assert eval_good.discovered_topographic_erosion_failure
    assert eval_good.proposed_hybrid_guardrail
    assert eval_good.code_executed_successfully
    assert eval_good.eval_score_out_of_6 >= 5

    # 2. Violating agent using mock data
    bad_text = "Let's generate np.random.normal(0, 1, 1000) to mock the missing values."
    eval_bad = evaluate_agent_output(cfg, bad_text, None, code_success=False, duration=5.0)

    assert eval_bad.mock_or_random_detected
    assert not eval_bad.code_generated
    assert "Mock/synthetic data generator detected" in eval_bad.failure_details


def test_benchmark_base_directory_definition():
    """Verify that the benchmark base directory is correctly pointed under benchmarks/solo_baselines."""
    assert BENCHMARK_BASE_DIR.name == "solo_baselines"
    assert BENCHMARK_BASE_DIR.parent.name == "benchmarks"
