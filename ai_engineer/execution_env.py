import subprocess
import time

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ai_engineer.filesdict import FilesDict


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int

class ExecutionEnv:
    def __init__(self, working_dir:str|Path):
        self.working_dir = Path(working_dir)
    
    def upload(self, files_dict:FilesDict):
        files_dict.to_file(self.working_dir)

        return self
    
    def popen(self, command: str) -> subprocess.Popen:
        p = subprocess.Popen(
            command,
            shell=True,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p

    def run(self, command:str, timeout:Optional[int]=None) -> ExecutionResult:
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
                    print(stdout.decode("utf-8"), end="")
                    print(stderr.decode("utf-8"), end="")
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
            print(stdout.decode("utf-8"), end="")
            print(stderr.decode("utf-8"), end="")
            print("\n--- Finished run ---\n")

        return ExecutionResult(stdout_full, stderr_full, p.returncode)


