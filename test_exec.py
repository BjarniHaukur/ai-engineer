from pathlib import Path

from ai_engineer.ai import AI
from ai_engineer.actions import run_code
from ai_engineer.filesdict import FilesDict
from ai_engineer.execution_env import ExecutionEnv

OUTPUT_DIR = Path("outputs")
EXEC_DIR = Path("exec")

files_dict = FilesDict.from_file(OUTPUT_DIR)

env = ExecutionEnv(EXEC_DIR)

#env.upload(files_dict)

stdout, stderr, returncode = env.run("uv run src/main.py")

print(stdout)
print(stderr)
print(returncode)

