# Dataset v5 Recovery Specification

## Purpose

Defines the requirements for replacing the v4 template-heavy data generation with a high-entropy v5 dataset to recover model performance and pass the evaluation hard-gate.

## Requirements

### Requirement: Generate High-Entropy Dataset

The system MUST generate a new dataset (`thinkanywhere_sft_v5.jsonl`) consisting of 150-200 unique examples with varying thinking styles and randomized structure.

#### Scenario: V5 Dataset Generation
- GIVEN the `scripts/generate_sft_v5.py` script is executed
- WHEN the dataset generation process completes
- THEN the output file `data/thinkanywhere_sft_v5.jsonl` MUST contain between 150 and 200 unique examples
- AND the thinking blocks across examples MUST exhibit randomized structure and vocabulary

### Requirement: Strict Dataset Quality Constraints

The system MUST enforce strict validation on generated code to reject lazy placeholders, ensuring every example contains a full, executable implementation.

#### Scenario: Code Placeholder Rejection
- GIVEN the dataset generator is producing an example
- WHEN the generated code contains `pass` or placeholder comments (e.g., `# implementation here`, `# TODO`)
- THEN the generator MUST reject the example and regenerate it until it contains valid, complete, and executable code

#### Scenario: Verification Step Injection
- GIVEN the dataset generator is producing an example
- WHEN constructing the thinking block
- THEN the generator SHOULD inject diverse verification steps or "negative" examples to prevent thought loops

### Requirement: Update Training Configuration

The system MUST update the training configuration to utilize the newly generated v5 dataset.

#### Scenario: Training Script Update
- GIVEN the `training/sft_lora.py` script
- WHEN executing the training pipeline
- THEN the script MUST point to `data/thinkanywhere_sft_v5.jsonl` as the training data source

### Requirement: Hard Gate Evaluation Pass

The system MUST pass the evaluation hard-gate to prove full recovery from Phase 4 failure.

#### Scenario: Hard Gate Recovery Validation
- GIVEN a model trained on the v5 dataset
- WHEN evaluated against the 30-problem held-out evaluation set
- THEN the model MUST score strictly greater than 2/30 (Phase 4 Base model score)
