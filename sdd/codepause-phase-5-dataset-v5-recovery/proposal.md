# Proposal: CodePause Phase 5 Dataset v5 Recovery

## Intent

Phase 4 failed the hard-gate (0/30) versus the Base model (2/30) due to Dataset v4 formatting collapse and logical laziness (e.g., using `pass` or `# implementation` instead of actual code). This proposal aims to recover model performance by replacing the template-heavy, repetitive data generation process with a high-entropy, diverse Dataset v5 that forces complete and correct implementations.

## Scope

### In Scope
- Create a new Dataset v5 generator (`scripts/generate_sft_v5.py`).
- Generate 150-200 unique examples with varying thinking styles and randomized structure.
- Ensure every generated example contains a full, functional implementation without `pass` or placeholders.
- Inject "negative" examples or diverse verification steps to prevent thought loops.
- Retrain the model using the new Dataset v5 (`training/sft_lora.py`).

### Out of Scope
- Using large teacher models (e.g., GPT-4/Claude) for synthetic data generation.
- Changing the underlying base model architecture.
- Adjusting training hyperparameters unrelated to data quality recovery.

## Approach

**Targeted Dataset v5 Recovery**:
We will implement Approach 1 from the exploration. The core issue is that the current generator trains the model to say "pass" and outputs identical patterns. We will update the generator logic to produce high-entropy, valid data. This involves randomizing the vocabulary and order of sections in the thinking block, enforcing strict validation on generated code to reject lazy placeholders, and ensuring diversity across 150-200 examples. We will carefully balance the dataset to prevent catastrophic forgetting while tuning the data quality.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `scripts/generate_sft_v5.py` | New | High-entropy dataset generator replacing v4. |
| `data/thinkanywhere_sft_v5.jsonl` | New | 150-200 unique examples with complete implementations. |
| `training/sft_lora.py` | Modified | Update to point to v5 data. |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Overfitting to the 30-problem set | Medium | Ensure dataset diversity and test against a separate validation set. |
| Catastrophic Forgetting | Medium | Monitor LoRA metrics; do not over-train on the small dataset. |

## Rollback Plan

If Phase 5 training yields degraded performance or fails to recover the hard-gate, we will revert the data generator and restore the `training/sft_lora.py` configuration to reference the last functional checkpoint, abandoning v5 data.

## Dependencies

- Existing 30-problem held-out evaluation set for the hard-gate.

## Success Criteria

- [ ] `generate_sft_v5.py` produces 150+ valid examples with no `pass` placeholders.
- [ ] Model trains successfully on v5 data without format collapse or repetitive loops.
- [ ] **Hard Gate Alignment**: Phase 5 model scores strictly > 2/30 on the hard-gate evaluation, proving full recovery from Phase 4 failure.