# Phase 2: Quality & Scale

## Overview
Phase 2 focuses on fixing the evaluation extraction logic and prompt alignment issues identified in Phase 1C, without introducing framework migrations or hardware upscaling.

## Objectives
- **Failure Taxonomy**: Document the root causes of the 0/30 Phase 1C failures.
- **Hardened Extraction Primitive**: Implement `extract_code()` with fallback parsing and `ast.parse` syntax validation.
- **Prompt Templates**: Introduce explicit boundaries and formatting strings in `src/prompt_templates.py`.
- **Focused Testing**: Verify extraction and templates thoroughly before scaling evaluation.