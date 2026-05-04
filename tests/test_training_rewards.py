class TestRewards:
    def test_module_imports_without_error(self):
        import training.rewards as rewards

        assert hasattr(rewards, "compute_rewards")

    def test_compute_rewards_returns_dict(self):
        import training.rewards as rewards

        result = rewards.compute_rewards("some generated code", {"id": "p1"})
        assert isinstance(result, dict)
        assert "total" in result
        assert isinstance(result["total"], float)

    def test_compute_rewards_with_passing_code(self):
        import training.rewards as rewards

        result = rewards.compute_rewards("def f(): pass", {"id": "p1"})
        assert result["total"] >= 0.0

    def test_compute_rewards_stub_values(self):
        import training.rewards as rewards

        result = rewards.compute_rewards("any code", {"id": "p1"})
        assert "accuracy" in result
        assert "format" in result
        assert "total" in result
