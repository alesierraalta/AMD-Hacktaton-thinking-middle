# Tasks: CodePause Phase 5 Dataset v5 Recovery

## Phase 1: Generator Foundation
- [x] 1.1 Create `scripts/generate_sft_v5.py` with CLI arguments and main entrypoint.
- [x] 1.2 Implement string-matching validation function to reject code containing `pass`, `# implementation`, or `# TODO`.
- [x] 1.3 Add dynamic thinking block assembler with randomized tags, headers, and verification phrasing.

## Phase 2: Core Dataset Generation
- [x] 2.1 Add parameterized algorithm templates (sorting, math, string ops) with diverse parameters.
- [x] 2.2 Write the synthesis loop to generate 150-200 valid examples into `data/thinkanywhere_sft_v5.jsonl`.
- [x] 2.3 Run `generate_sft_v5.py` and verify `thinkanywhere_sft_v5.jsonl` contains valid, full implementations.

## Phase 3: Training Script Update
- [x] 3.1 Update `training/sft_lora.py` `_normalize_sft_record` to support `prompt`/`raw_output` schemas.
- [x] 3.2 Change default dataset path in `training/sft_lora.py` to `data/thinkanywhere_sft_v5.jsonl`.
- [x] 3.3 Run `python training/sft_lora.py --dry-run` to ensure successful parsing of the v5 dataset.

## Phase 4: Training & Evaluation
- [ ] 4.1 Run the full fine-tuning pipeline with `sft_lora.py` using the new dataset.
- [ ] 4.2 Evaluate the trained model on the 30-problem evaluation set to confirm the score > 2/30.
