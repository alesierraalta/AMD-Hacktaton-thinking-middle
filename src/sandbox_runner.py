import subprocess
import sys
import tempfile
import os


def run_code(code: str, tests: list[str], timeout: int = 5) -> dict:
    script = code + "\n"
    for test in tests:
        script += test + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "returncode": -1,
            "timeout": True,
        }
    finally:
        os.unlink(temp_path)
