import subprocess
import sys
import os
import time
from collections import deque
from typing import Union, List, Optional, Dict

def run_step(
    name: str,
    cmd: Union[str, List[str]],
    cwd: Optional[str] = None,
    log_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tail_lines: int = 40
) -> str:
    """
    Executes a command with real-time output streaming and logging.
    
    Args:
        name: Name of the step for visual feedback.
        cmd: Command to execute (string for shell, list for direct).
        cwd: Working directory.
        log_path: Path to save the output logs.
        env: Environment variables.
        tail_lines: Number of lines to keep for error reporting on failure.
        
    Returns:
        The complete output of the command.
        
    Raises:
        RuntimeError: If the command returns a non-zero exit code.
    """
    print(f"\n>>> [STEP: {name}] Executing: {cmd}")
    
    full_output = []
    tail_buffer = deque(maxlen=tail_lines)
    
    # Ensure log directory exists
    log_file = None
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    shell = isinstance(cmd, str)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=shell,
            cwd=cwd,
            env=env or os.environ.copy(),
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output line by line
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            
            # Print to stdout immediately
            print(line, end="", flush=True)
            sys.stdout.flush()
            
            full_output.append(line)
            tail_buffer.append(line)
            
            if log_file:
                log_file.write(line)
                log_file.flush()
                
        process.stdout.close()
        return_code = process.wait()
        
    except KeyboardInterrupt:
        print(f"\n!!! [STEP: {name}] Interrupted by user.")
        if 'process' in locals():
            process.terminate()
        raise
    finally:
        if log_file:
            log_file.close()

    result_text = "".join(full_output)
    
    if return_code != 0:
        error_tail = "".join(tail_buffer)
        error_msg = (
            f"\n\n[HARD STOP] Step '{name}' failed with return code {return_code}.\n"
            f"--- [LAST {tail_lines} LINES OF OUTPUT] ---\n"
            f"{error_tail}\n"
            f"--- [END OF TAIL] ---\n"
        )
        raise RuntimeError(error_msg)
        
    print(f">>> [STEP: {name}] Completed successfully.\n")
    return result_text
