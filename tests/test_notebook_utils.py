import unittest
import os
import shutil
import tempfile
import sys
from app.notebook_utils import run_step

class TestNotebookUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_step_success(self):
        output = run_step("TestSuccess", [sys.executable, "-c", "print('hello world')"])
        self.assertIn("hello world", output)

    def test_run_step_shell_success(self):
        output = run_step("TestShellSuccess", f'"{sys.executable}" -c "print(\'shell works\')"')
        self.assertIn("shell works", output)

    def test_run_step_failure(self):
        with self.assertRaises(RuntimeError) as cm:
            run_step("TestFailure", [sys.executable, "-c", "print('first line'); print('second line'); raise SystemExit(1)"])
        
        self.assertIn("[HARD STOP]", str(cm.exception))
        self.assertIn("first line", str(cm.exception))
        self.assertIn("second line", str(cm.exception))

    def test_run_step_logging(self):
        log_path = os.path.join(self.test_dir, "subdir/test.log")
        run_step("TestLogging", [sys.executable, "-c", "print('log this')"], log_path=log_path)
        
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r") as f:
            content = f.read()
            self.assertIn("log this", content)

if __name__ == "__main__":
    unittest.main()
