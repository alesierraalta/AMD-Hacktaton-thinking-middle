# MODEL CARD - CodePause RC1

## Model Details
- **Base Model**: `Qwen/Qwen1.5-1.8B-Chat`
- **Adapter Type**: LoRA / QLoRA (4-bit)
- **Language(s)**: Python (primary), English (reasoning)
- **License**: Apache 2.0 (matches base model)

## Intended Use
- **Primary Use**: Research into internal reasoning blocks for code generation.
- **Out of Scope**: Production-critical code generation, security-sensitive applications.

## Training Hardware
- **GPU**: 1x NVIDIA Tesla T4 (Google Colab)
- **VRAM**: ~15GB used during training (QLoRA)

## Evaluation Results
- **Dataset**: `heldout_problems_30.jsonl` (30 algorithmic problems)
- **Metric**: Pass@1
- **Score**: 10.0% (3/30 passed)
- **Baseline Comparison**: 3.3% (1+2 improvement)

## Loading Instructions
To load the model on a T4 or equivalent GPU:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
adapter_path = "results/phase6_release_candidate/"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
```

## Limitations
- **Model Size**: The 1.8B parameter count limits the complexity of reasoning.
- **Accuracy**: 90% of held-out problems still fail, mostly due to execution errors or logic bugs in the generated code.
- **Hardware**: Validated only on NVIDIA hardware. Not tested on AMD ROCm for this release.
- **Formatting**: The model occasionally generates malformed tags if the response is truncated.
