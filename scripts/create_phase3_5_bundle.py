"""Create a Colab overlay bundle for CodePause Phase 3.5 files.

Use when Phase 3.5 files are local/uncommitted and Colab clones the GitHub repo
without them. Upload the generated zip to `/content/phase3_5_bundle.zip` or to
`/content/drive/MyDrive/codepause_phase3_5/phase3_5_bundle.zip`.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase3_5_bundle.zip"

FILES = [
    "src/prompt_templates.py",
    "data/thinkanywhere_sft_v3.jsonl",
    "results/phase3_5_dataset_v3_quality_report.md",
    "config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml",
    "notebooks/codepause_phase_3_5_prompt_data_refinement.ipynb",
    "tests/test_phase3_prompt_template.py",
    "tests/test_phase3_evaluator_prompt_support.py",
    "tests/test_phase3_dataset_v3.py",
]

DIRS = [
    "sdd/codepause-phase-3-5-prompt-data-refinement",
]


def iter_paths() -> list[Path]:
    paths: list[Path] = []
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 3.5 file: {rel}")
        paths.append(path)

    for rel_dir in DIRS:
        path = ROOT / rel_dir
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 3.5 directory: {rel_dir}")
        paths.extend(p for p in path.rglob("*") if p.is_file())

    return sorted(paths)


def main() -> None:
    paths = iter_paths()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, path.relative_to(ROOT).as_posix())
    print(f"Created {OUT} with {len(paths)} files")


if __name__ == "__main__":
    main()
