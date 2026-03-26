import subprocess
import os
import platform
import logging
from typing import Dict, Any

logger = logging.getLogger("CodeExecutor")

def sanitize_code(code: str) -> str:
    """
    Strips markdown artifacts and prepares raw code for execution.
    """
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if len(lines) > 2:
            # Remove the first and last lines (markdown backticks)
            code = "\n".join(lines[1:-1])
    return code

def run_code(file_path: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Executes an artifact using the native host interpreter.
    Matches the API expected by xyber_env.py.
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": "File not found"}

    host_os = platform.system()
    ext = os.path.splitext(file_path)[1].lower()
    cmd = []

    # ── Interpreter Routing Logic (Windows/Linux/Mac) ──
    if ext == ".py":
        cmd = ["python", file_path]
    elif ext == ".ps1":
        if host_os == "Windows":
            # Bypass execution policy to ensure the agent isn't blocked by OS defaults
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", file_path]
        else:
            # Support for PowerShell Core on Linux/macOS
            cmd = ["pwsh", "-File", file_path]
    elif ext == ".sh":
        if host_os == "Windows":
            # Fallback to 'sh' (likely Git Bash) if available in PATH
            cmd = ["sh", file_path]
        else:
            cmd = ["bash", file_path]
    else:
        # Direct execution for binaries or batch files
        cmd = [file_path]

    try:
        # Using list-based execution for safety and output capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": False
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"Execution timed out: {file_path}")
        return {"success": False, "error": "Timeout", "timed_out": True}
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return {"success": False, "error": str(e)}