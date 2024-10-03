from pathlib import Path

from ai_engineer.ai import AI
from ai_engineer.actions import run_code
from ai_engineer.filesdict import FilesDict
from ai_engineer.execution_env import ExecutionEnv

OUTPUT_DIR = Path("outputs")
EXEC_DIR = Path("exec")

files_dict = FilesDict.from_file(OUTPUT_DIR)

ai = AI(model_name="gpt-4o")

env = ExecutionEnv(EXEC_DIR)

output = run_code(ai, env, files_dict)

# #print(exec_command)
print(output)
