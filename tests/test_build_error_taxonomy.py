from pathlib import Path

from src.evaluation.build_error_taxonomy import analyse_traces, render_markdown


def test_taxonomy_includes_required_categories_and_soft_skill_limitation() -> None:
    analysis = analyse_traces(
        [
            {"error_category": "boundary_error", "role_type": "software_dev"},
            {"error_category": "esco_coverage_gap", "role_type": "qa_engineer"},
            {"error_category": "soft_skill_ambiguity", "role_type": "data_analyst"},
            {"error_category": "false_positive", "role_type": "software_dev"},
            {"error_category": "correct", "role_type": "software_dev"},
        ]
    )

    assert analysis["counts"] == {"E1": 1, "E2": 1, "E3": 1, "E5": 1}
    assert analysis["error_records"] == 4
    assert analysis["frequencies"]["E1"] == 0.25
    assert "Soft-Skill Ambiguity" in render_markdown(analysis)


def test_render_markdown_reports_zero_soft_skill_cases() -> None:
    analysis = analyse_traces([{"error_category": "boundary_error", "role_type": "software_dev"}])

    markdown = render_markdown(analysis)

    assert "| E3 | Soft-Skill Ambiguity | 0 | 0.00% |" in markdown
    assert "no reviewed gold spans were labelled `SKILL_TRANS`" in markdown
