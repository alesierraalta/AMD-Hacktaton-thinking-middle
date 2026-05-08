"""Tests for Main Evaluation pipeline enhancements in codepause_phase_2_quality_scale.ipynb.

These tests validate the 4 notebook-only enhancements per the SDD design:
1. Metadata provenance generation (phase, model, hardware, timestamp, git_sha, notebook)
2. Mock dry-run cell (validates CLI wiring before expensive GPU eval)
3. Pre-flight dataset validation (validates 30 rows, required keys)
4. Inline report display (IPython.display.Markdown after make_report calls)
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path('notebooks/codepause_phase_2_quality_scale.ipynb')


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8-sig'))


def notebook_sources():
    """Return all code cell sources joined."""
    nb = load_notebook()
    return [''.join(cell.get('source', [])) for cell in nb.get('cells', []) if cell.get('cell_type') == 'code']


def all_sources():
    return '\n'.join(notebook_sources())


class TestMockDryRunCell:
    """Cell 1c: Mock dry-run validates CLI args + wiring before GPU eval."""

    def test_mock_dry_run_cell_exists_after_setup(self):
        sources = notebook_sources()
        # Should appear after setup cell (index 1)
        # Find the setup cell ending
        setup_idx = None
        for i, src in enumerate(sources):
            if 'subprocess.run' in src and 'BRANCH' in src:
                setup_idx = i
                break
        assert setup_idx is not None, "Setup cell not found"
        # Metadata must be generated before the mock because the mock wires --metadata_json.
        assert len(sources) > setup_idx + 1, "No cell after setup"
        mock_src = sources[setup_idx + 2]
        assert '--mock' in mock_src, "Mock cell must use --mock flag"

    def test_mock_cell_calls_evaluate_baseline(self):
        sources = notebook_sources()
        mock_src = '\n'.join(sources)
        assert 'evaluate_baseline.py' in mock_src, "Mock must call evaluate_baseline.py"

    def test_mock_cell_uses_sanity_problems(self):
        sources = notebook_sources()
        mock_src = '\n'.join(sources)
        assert 'sanity_problems.jsonl' in mock_src, "Mock must use sanity_problems.jsonl"

    def test_mock_cell_passes_metadata_json(self):
        sources = notebook_sources()
        mock_src = '\n'.join(sources)
        # Should have --metadata_json flag
        assert '--metadata_json' in mock_src, "Mock cell should pass --metadata_json for wiring validation"


class TestPreFlightValidationCell:
    """Cell 1d: Pre-flight dataset validation using validate_dataset.py."""

    def test_preflight_cell_exists_after_mock(self):
        sources = notebook_sources()
        # Find mock cell
        mock_idx = None
        for i, src in enumerate(sources):
            if '--mock' in src and 'evaluate_baseline.py' in src:
                mock_idx = i
                break
        assert mock_idx is not None, "Mock cell not found"
        assert len(sources) > mock_idx + 1, "No cell after mock"
        preflight_src = sources[mock_idx + 1]
        assert 'validate_dataset.py' in preflight_src, "Pre-flight must call validate_dataset.py"

    def test_preflight_validates_problems_schema(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert '--schema' in joined and 'problems' in joined, "Must validate --schema problems"

    def test_preflight_checks_exactly_30_examples(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        # Should have min_examples=30 and max_examples=30
        assert '--min_examples' in joined and '30' in joined, "Must check min 30 examples"
        assert '--max_examples' in joined and '30' in joined, "Must check max 30 examples"

    def test_preflight_validates_main_problems_dataset(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'data/problems.jsonl' in joined, "Pre-flight must validate data/problems.jsonl"


class TestMetadataGenerationCell:
    """Cell 1b: Metadata provenance generation for results/metadata_run.json."""

    def test_metadata_generation_cell_exists(self):
        sources = notebook_sources()
        # Find setup cell; metadata must be immediately after setup.
        setup_idx = None
        for i, src in enumerate(sources):
            if 'subprocess.run' in src and 'BRANCH' in src:
                setup_idx = i
                break
        assert setup_idx is not None, "Setup cell not found"
        assert len(sources) > setup_idx + 1, "No cell after setup"
        metadata_src = sources[setup_idx + 1]
        # Should write metadata JSON
        assert 'metadata_run.json' in metadata_src, "Metadata cell must reference metadata_run.json"

    def test_metadata_contains_phase_field(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert '"phase"' in joined or "'phase'" in joined, "Metadata must include phase field"

    def test_metadata_contains_model_name(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'model_name' in joined, "Metadata must include model_name"

    def test_metadata_contains_hardware_detection(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        # Should detect GPU via torch.cuda.get_device_name or fallback to cpu
        assert ('torch.cuda' in joined or 'nvidia-smi' in joined or 'GPU' in joined or 'hardware' in joined.lower()), \
            "Metadata must include hardware detection"

    def test_metadata_contains_timestamp(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'timestamp' in joined or 'datetime' in joined or 'time' in joined, "Metadata must include timestamp"

    def test_metadata_contains_git_sha(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'git_sha' in joined or 'git rev-parse' in joined or 'SHA' in joined, "Metadata must include git_sha"


class TestEvalCellsMetadataInjection:
    """4 evaluation cells should include --metadata_json flag."""

    def test_baseline_sanity_eval_has_metadata_flag(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        # Find baseline sanity eval
        assert 'phase2_sanity_baseline.jsonl' in joined, "Baseline sanity eval not found"
        # The cell should include both the output path AND the --metadata_json flag
        sanity_idx = None
        for i, src in enumerate(sources):
            if 'phase2_sanity_baseline.jsonl' in src and 'evaluate_baseline.py' in src:
                sanity_idx = i
                break
        assert sanity_idx is not None
        src = sources[sanity_idx]
        assert '--metadata_json' in src, "Baseline sanity eval must include --metadata_json"

    def test_baseline_main_eval_has_metadata_flag(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'phase2_main_baseline.jsonl' in joined, "Baseline main eval not found"
        main_idx = None
        for i, src in enumerate(sources):
            if 'phase2_main_baseline.jsonl' in src and 'evaluate_baseline.py' in src:
                main_idx = i
                break
        assert main_idx is not None
        src = sources[main_idx]
        assert '--metadata_json' in src, "Baseline main eval must include --metadata_json"

    def test_finetuned_sanity_eval_has_metadata_flag(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'phase2_sanity_finetuned.jsonl' in joined, "Fine-tuned sanity eval not found"
        ft_sanity_idx = None
        for i, src in enumerate(sources):
            if 'phase2_sanity_finetuned.jsonl' in src and 'evaluate_finetuned.py' in src:
                ft_sanity_idx = i
                break
        assert ft_sanity_idx is not None
        src = sources[ft_sanity_idx]
        assert '--metadata_json' in src, "Fine-tuned sanity eval must include --metadata_json"

    def test_finetuned_main_eval_has_metadata_flag(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        assert 'phase2_main_finetuned.jsonl' in joined, "Fine-tuned main eval not found"
        ft_main_idx = None
        for i, src in enumerate(sources):
            if 'phase2_main_finetuned.jsonl' in src and 'evaluate_finetuned.py' in src:
                ft_main_idx = i
                break
        assert ft_main_idx is not None
        src = sources[ft_main_idx]
        assert '--metadata_json' in src, "Fine-tuned main eval must include --metadata_json"

    def test_metadata_json_path_is_consistent(self):
        sources = notebook_sources()
        joined = '\n'.join(sources)
        # All --metadata_json should point to results/metadata_run.json
        # Pattern may span multiple lines in the command list
        import re
        # Match --metadata_json followed by the path (could be on same or next line)
        matches = re.findall(r'--metadata_json",\s*"([^"]+)"', joined)
        if len(matches) < 4:
            # Also try multiline pattern where path is on separate list item
            matches = re.findall(r'--metadata_json[\s,]*"?([^"\n]+?)"?', joined)
        assert len(matches) >= 4, f"Expected at least 4 --metadata_json calls, found {len(matches)}: {matches}"
        for m in matches:
            assert 'metadata_run.json' in m, f"All metadata_json should use results/metadata_run.json, found: {m}"


class TestInlineReportDisplay:
    """After make_report.py calls, inline Markdown display should render the report."""

    def test_sanity_report_has_markdown_display(self):
        sources = notebook_sources()
        # Find the cell with sanity report generation
        sanity_report_idx = None
        for i, src in enumerate(sources):
            if 'phase2_sanity_report.md' in src and 'make_report.py' in src:
                sanity_report_idx = i
                break
        assert sanity_report_idx is not None, "Sanity report generation not found"
        src = sources[sanity_report_idx]
        # Should have Markdown display after the report
        assert ('Markdown' in src or 'IPython' in src or 'display(Markdown' in src), \
            "Sanity report cell should include inline Markdown display"

    def test_main_report_has_markdown_display(self):
        sources = notebook_sources()
        # Find the cell with main report generation
        main_report_idx = None
        for i, src in enumerate(sources):
            if 'phase2_main_report.md' in src and 'make_report.py' in src:
                main_report_idx = i
                break
        assert main_report_idx is not None, "Main report generation not found"
        src = sources[main_report_idx]
        # Should have Markdown display after the report
        assert ('Markdown' in src or 'IPython' in src or 'display(Markdown' in src), \
            "Main report cell should include inline Markdown display"


class TestPipelineOrder:
    """Validate the pipeline cell ordering: Setup → Metadata → Mock → Pre-flight → Evals → Reports."""

    def test_cell_order_metadata_before_mock(self):
        sources = notebook_sources()
        metadata_idx = mock_idx = None
        for i, src in enumerate(sources):
            if 'metadata_run.json' in src and 'torch.cuda' in src and 'validate_dataset.py' not in src:
                metadata_idx = i
            if '--mock' in src and 'evaluate_baseline.py' in src:
                mock_idx = i
        assert metadata_idx is not None and mock_idx is not None, \
            f"metadata_idx={metadata_idx}, mock_idx={mock_idx}"
        assert metadata_idx < mock_idx, "Metadata must be generated before mock uses --metadata_json"

    def test_cell_order_mock_before_preflight(self):
        sources = notebook_sources()
        mock_idx = preflight_idx = None
        for i, src in enumerate(sources):
            if '--mock' in src and 'evaluate_baseline.py' in src:
                mock_idx = i
            if 'validate_dataset.py' in src:
                preflight_idx = i
        assert mock_idx is not None and preflight_idx is not None
        assert mock_idx < preflight_idx, "Mock should come before pre-flight"

    def test_cell_order_metadata_before_preflight(self):
        sources = notebook_sources()
        preflight_idx = metadata_idx = None
        for i, src in enumerate(sources):
            if 'validate_dataset.py' in src:
                preflight_idx = i
            if 'metadata_run.json' in src and 'torch.cuda' in src and 'validate_dataset.py' not in src:
                metadata_idx = i
        assert preflight_idx is not None and metadata_idx is not None, \
            f"preflight_idx={preflight_idx}, metadata_idx={metadata_idx}"
        assert metadata_idx < preflight_idx, "Metadata should come before pre-flight and eval cells"

    def test_cell_order_metadata_before_evals(self):
        sources = notebook_sources()
        metadata_idx = baseline_main_idx = None
        for i, src in enumerate(sources):
            # Find metadata generation cell (has torch.cuda and writes metadata_run.json)
            if 'metadata_run.json' in src and 'torch.cuda' in src and 'validate_dataset.py' not in src:
                metadata_idx = i
            # Find main baseline eval cell (phase2_main_baseline, NOT sanity)
            if 'phase2_main_baseline.jsonl' in src and 'evaluate_baseline.py' in src and 'sanity' not in src:
                baseline_main_idx = i
        assert metadata_idx is not None and baseline_main_idx is not None, \
            f"metadata_idx={metadata_idx}, baseline_main_idx={baseline_main_idx}"
        assert metadata_idx < baseline_main_idx, f"Metadata (cell {metadata_idx}) should come before main eval (cell {baseline_main_idx})"
