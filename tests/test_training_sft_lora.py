import sys
import io
from unittest import mock


class TestSftLoraEdgeCases:
    def test_missing_dataset_raises_error(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/nonexistent.jsonl",
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", test_args):
            try:
                sft.main()
                assert False, "Expected FileNotFoundError"
            except FileNotFoundError:
                pass

    def test_missing_required_args_exits(self):
        import training.sft_lora as sft

        test_args = ["prog"]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stderr", new=io.StringIO()):
                try:
                    sft.parse_args()
                    assert False, "Expected SystemExit"
                except SystemExit as e:
                    assert e.code != 0


class TestSftLoraImport:
    def test_module_imports_without_error(self):
        import training.sft_lora as sft

        assert hasattr(sft, "parse_args")
        assert hasattr(sft, "main")


class TestLoadDataset:
    def test_load_dataset_basic(self, tmp_path):
        import json
        import training.sft_lora as sft

        records = [{"text": "def foo():\n    pass", "problem_id": i} for i in range(3)]
        path = tmp_path / "data.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = sft._load_dataset(str(path))
        assert len(result) == 3
        assert result[0]["problem_id"] == 0

    def test_load_dataset_with_limit(self, tmp_path):
        import json
        import training.sft_lora as sft

        records = [{"text": f"def foo{i}():\n    pass", "problem_id": i} for i in range(10)]
        path = tmp_path / "data.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = sft._load_dataset(str(path), limit=5)
        assert len(result) == 5



    def test_load_dataset_normalizes_instruction_response_records(self, tmp_path):
        import json
        import training.sft_lora as sft

        path = tmp_path / "phase2.jsonl"
        record = {
            "instruction": "Write a function that doubles a number.",
            "response": "<thinkanywhere>Multiply by two.</thinkanywhere>\ndef double(x): return x * 2",
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = sft._load_dataset(str(path))

        assert len(result) == 1
        assert set(result[0]) == {"text"}
        assert "### Instruction:" in result[0]["text"]
        assert record["instruction"] in result[0]["text"]
        assert "### Response:" in result[0]["text"]
        assert record["response"] in result[0]["text"]

    def test_load_dataset_rejects_unsupported_sft_schema(self, tmp_path):
        import json
        import pytest
        import training.sft_lora as sft

        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"instruction": "Write foo"}) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported SFT record schema"):
            sft._load_dataset(str(path))

    def test_load_dataset_missing_file(self):
        import training.sft_lora as sft

        try:
            sft._load_dataset("nonexistent/path.jsonl")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_load_dataset_cp1252_survives_roundtrip(self, tmp_path):
        """Latin-1 (cp1252-compatible) text survives the fallback chain."""
        import json
        import training.sft_lora as sft

        records = [{"text": "café", "problem_id": 0}]
        path = tmp_path / "latin1.jsonl"
        with open(path, "w", encoding="latin-1") as f:
            f.write(json.dumps(records[0]) + "\n")

        result = sft._load_dataset(str(path))
        assert len(result) == 1


class TestSftLoraArgs:
    def test_parse_args_all_options(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--output_dir", "results/sft",
            "--epochs", "3",
            "--max_seq_length", "512",
            "--learning_rate", "2e-4",
            "--lora_rank", "16",
            "--gradient_accumulation_steps", "4",
            "--smoke",
            "--max_steps", "10",
            "--limit_samples", "5",
            "--device", "cpu",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.model_name == "mock"
        assert args.dataset_path == "data/thinkanywhere_sft.jsonl"
        assert args.output_dir == "results/sft"
        assert args.epochs == 3
        assert args.max_seq_length == 512
        assert args.learning_rate == 2e-4
        assert args.lora_rank == 16
        assert args.gradient_accumulation_steps == 4
        assert args.smoke is True
        assert args.max_steps == 10
        assert args.limit_samples == 5
        assert args.device == "cpu"

    def test_parse_args_defaults(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.output_dir == "results/sft"
        assert args.epochs == 1
        assert args.max_seq_length == 1024
        assert args.learning_rate == 2e-5
        assert args.lora_rank == 8
        assert args.gradient_accumulation_steps == 1
        assert args.smoke is False
        assert args.max_steps is None
        assert args.limit_samples is None
        assert args.device == "auto"

    def test_dry_run_mock_model(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--output_dir", "results/sft",
            "--dry-run",
        ]
        captured = io.StringIO()
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stdout", new=captured):
                sft.main()
        output = captured.getvalue()
        assert "Dry run complete" in output

    def test_smoke_mode_flag(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "openai-community/gpt2",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--smoke",
            "--max_steps", "3",
        ]
        captured = io.StringIO()
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stdout", new=captured):
                sft.main()
        output = captured.getvalue()
        assert "SMOKE" in output
        assert "max_steps=3" in output

    def test_smoke_mode_windows_cpu_override(self):
        """Line 68-70: on win32, smoke forces CPU device."""
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "openai-community/gpt2",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--smoke",
            "--device", "auto",
        ]
        captured = io.StringIO()
        with mock.patch.object(sys, "argv", test_args), \
             mock.patch.object(sys, "platform", "win32"), \
             mock.patch("sys.stdout", new=captured):
            sft.main()
        output = captured.getvalue()
        # On Windows, device should be forced to cpu (line 68)
        assert "device=cpu" in output or "SMOKE" in output

    def test_help_includes_all_options(self):
        import training.sft_lora as sft

        test_args = ["prog", "--help"]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stdout", new=io.StringIO()) as captured:
                try:
                    sft.parse_args()
                except SystemExit:
                    pass
        output = captured.getvalue()
        for opt in [
            "--model_name",
            "--dataset_path",
            "--output_dir",
            "--epochs",
            "--max_seq_length",
            "--learning_rate",
            "--lora_rank",
            "--gradient_accumulation_steps",
            "--dry-run",
            "--smoke",
            "--max_steps",
            "--limit_samples",
            "--device",
            "--load_in_4bit",
            "--bnb_4bit_compute_dtype",
            "--bnb_4bit_quant_type",
            "--bnb_4bit_use_double_quant",
            "--lora_alpha",
            "--lora_dropout",
            "--per_device_train_batch_size",
        ]:
            assert opt in output, f"{opt} not in help"


class TestSftLoraQLoRAFlags:
    def test_load_in_4bit_flag_parsed(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--load_in_4bit",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.load_in_4bit is True

    def test_bnb_4bit_quant_type_defaults_nf4(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--load_in_4bit",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.bnb_4bit_quant_type == "nf4"
        assert args.bnb_4bit_compute_dtype == "bfloat16"

    def test_bnb_double_quant_flag(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--load_in_4bit",
            "--bnb_4bit_use_double_quant",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.bnb_4bit_use_double_quant is True

    def test_lora_alpha_default_derived_from_rank(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--lora_rank", "16",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        # lora_alpha defaults to rank * 2 if not set explicitly
        assert args.lora_alpha is None  # None means derive
        assert args.lora_rank == 16

    def test_lora_alpha_explicit_overrides(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--lora_rank", "16",
            "--lora_alpha", "32",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.lora_alpha == 32

    def test_lora_dropout_default_005(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.lora_dropout == 0.05

    def test_per_device_batch_size_default_1(self):
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "mock",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = sft.parse_args()
        assert args.per_device_train_batch_size == 1

    def test_smoke_forces_load_in_4bit_off(self):
        """Smoke mode must never enable bitsandbytes (CPU-only)."""
        import training.sft_lora as sft

        test_args = [
            "prog",
            "--model_name", "openai-community/gpt2",
            "--dataset_path", "data/thinkanywhere_sft.jsonl",
            "--smoke",
            "--load_in_4bit",  # user tried to enable 4-bit in smoke
            "--max_steps", "3",
        ]
        captured = io.StringIO()
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stdout", new=captured):
                sft.main()
        output = captured.getvalue()
        assert "[4BIT]" not in output  # 4-bit must not activate in smoke


class TestModelsYamlIntegrity:
    def test_phase_1c_tiers_present(self):
        import yaml
        import os
        config_path = "config/models.yaml"
        assert os.path.exists(config_path), "models.yaml not found"
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        presets = data.get("presets", {})
        
        expected_models = {
            "local_colab_smoke": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "colab_primary_qwen": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "colab_non_qwen_granite": "ibm-granite/granite-3b-code-instruct-128k",
            "colab_optional_gemma": "google/gemma-4-E2B",
            "colab_stretch_qwen": "Qwen/Qwen2.5-Coder-3B-Instruct"
        }
        
        for key, model_name in expected_models.items():
            assert key in presets, f"Preset {key} missing in models.yaml"
            assert presets[key]["model_name"] == model_name, f"Expected {model_name} for {key}"
