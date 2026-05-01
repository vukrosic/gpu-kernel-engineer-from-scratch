from pathlib import Path


def test_results_scaffold_is_complete():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"

    week_files = sorted(results.glob("week-*.md"))
    month_files = sorted(results.glob("month-*.md"))

    assert len(week_files) == 48
    assert len(month_files) == 12
    assert (results / "benchmark-dashboard.md").exists()
    assert (results / "gpu-test-matrix.md").exists()
    assert (results / "figures").is_dir()

    sample = (results / "week-01-baseline.md").read_text(encoding="utf-8")
    assert "Status: scaffolded" in sample
