import json
from pathlib import Path

NOTEBOOKS = [
    Path('notebooks/codepause_colab_t4_qlora_probe.ipynb'),
    Path('notebooks/codepause_phase_1c_colab_only_qlora.ipynb'),
    Path('notebooks/codepause_phase_2_quality_scale.ipynb'),
]


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding='utf-8-sig'))
    return '\n'.join(
        ''.join(cell.get('source', []))
        for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code'
    )


def test_colab_notebooks_define_line_stream_runner():
    for path in NOTEBOOKS:
        source = notebook_source(path)
        assert 'subprocess.Popen' in source, f'{path} must stream command output with subprocess.Popen'
        assert 'stdout=subprocess.PIPE' in source, f'{path} must capture stdout for streaming'
        assert 'stderr=subprocess.STDOUT' in source, f'{path} must merge stderr into the visible stream'
        assert 'bufsize=1' in source, f'{path} must request line-buffered streaming'
        assert 'flush=True' in source or 'sys.stdout.flush()' in source, f'{path} must flush streamed output'


def test_phase2_runtime_gates_do_not_use_raw_python_bangs():
    source = notebook_source(Path('notebooks/codepause_phase_2_quality_scale.ipynb'))
    assert '!python ' not in source, 'Phase 2 runtime gates must use the streaming runner, not raw !python cells'
    for expected in ['eval/evaluate_baseline.py', 'eval/adapter_probe.py', 'eval/evaluate_finetuned.py']:
        assert expected in source



def test_phase2_evaluation_commands_include_output_paths():
    source = notebook_source(Path('notebooks/codepause_phase_2_quality_scale.ipynb'))
    assert '--output_path' in source
    assert 'results/phase2_sanity_baseline.jsonl' in source
    assert 'results/phase2_sanity_finetuned.jsonl' in source



def test_phase2_notebook_uses_real_recipe_a_training_and_adapter_path():
    source = notebook_source(Path('notebooks/codepause_phase_2_quality_scale.ipynb'))
    assert 'Ready for training with Recipe A' not in source
    assert 'training/sft_lora.py' in source
    assert 'data/thinkanywhere_sft_v2.jsonl' in source
    assert 'outputs/phase2_recipe_a/final_checkpoint' not in source
    assert 'outputs/phase2_recipe_a' in source
    assert 'Qwen/Qwen1.5-1.8B-Chat' in source


def test_phase2_setup_installs_training_dependencies_and_refreshes_repo():
    source = notebook_source(Path('notebooks/codepause_phase_2_quality_scale.ipynb'))
    assert 'trl' in source
    assert 'git fetch origin' in source
    assert 'git reset --hard origin/' in source



def test_colab_probe_uses_current_cli_flags():
    source = notebook_source(Path('notebooks/codepause_colab_t4_qlora_probe.ipynb'))
    assert '--metadata_json' in source
    assert '--metadata_path' not in source
    assert '--baseline' in source
    assert '--finetuned' in source
    assert '--out' in source
    assert '--baseline_path' not in source
    assert '--finetuned_path' not in source
    assert '--output_path' in source
