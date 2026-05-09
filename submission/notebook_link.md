# Notebook Links

The following notebooks contain the implementation, training, and evaluation logic for the CodePause submission.

## Primary Notebook (Phase 5 & 6)
- **Recovery & Release**: `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb`
  - *Purpose*: Dataset v5 generation, SFT training on T4, and final evaluation against the 30-problem test set.

## Historical/Reference Notebooks
- **Phase 4 Failure**: `notebooks/codepause_phase_4_dataset_v4_laziness_loop.ipynb`
  - *Purpose*: Documents the failure mode of unconstrained synthetic data and the "laziness loop" behavior.
- **Phase 1C Initial Probe**: `notebooks/codepause_phase_1c_colab_only_qlora.ipynb`
  - *Purpose*: Initial Colab setup and quantization tests.

## Access Note
All notebooks are provided in the repository. They are configured to run in Google Colab T4 with standard libraries (Transformers, PEFT, TRL).
