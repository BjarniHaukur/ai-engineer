import time
import subprocess

from pathlib import Path
from typing import Optional, Tuple


class ExecutionEnv:
    def __init__(self, working_dir:str|Path, stdout:bool=False, stderr:bool=False):
        self.working_dir = Path(working_dir)
        self.stdout, self.stderr = stdout, stderr
    
    def popen(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            command,
            shell=True,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        

    def run(self, command:str, timeout:Optional[int]=None) -> Tuple[str, str, int]:
        start = time.time()
        
        p = self.popen(command)
        
        stdout_full, stderr_full = "", ""
        
        try:
            while p.poll() is None:
                if timeout and time.time() - start > timeout:
                    print("Timeout!")
                    p.kill()
                    raise TimeoutError()
                
                assert p.stdout is not None
                assert p.stderr is not None
                
                # Use communicate with a short timeout to read available output
                try:
                    stdout, stderr = p.communicate(timeout=0.1)
                    stdout_full += stdout.decode("utf-8")
                    stderr_full += stderr.decode("utf-8")
                    if self.stdout: print(stdout.decode("utf-8"), end="")
                    if self.stderr: print(stderr.decode("utf-8"), end="")
                except subprocess.TimeoutExpired:
                    # If communicate times out, continue the loop
                    continue
                
        except KeyboardInterrupt:
            print("\nStopping execution.")
            p.kill()
            print("Execution stopped.")
        finally:
            # Ensure we capture any remaining output
            stdout, stderr = p.communicate()
            stdout_full += stdout.decode("utf-8")
            stderr_full += stderr.decode("utf-8")
            if self.stdout: print(stdout.decode("utf-8"), end="")
            if self.stderr: print(stderr.decode("utf-8"), end="")
            print("\n--- Finished run ---\n")

        return stdout_full, stderr_full, p.returncode


