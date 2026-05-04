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
        ]:
            assert opt in output, f"{opt} not in help"
