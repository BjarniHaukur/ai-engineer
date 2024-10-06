from pathlib import Path

from ai_engineer.ai import AI
from ai_engineer.actions import run_code, fix_code
from ai_engineer.filesdict import FilesDict
from ai_engineer.execution_env import ExecutionEnv

OUTPUT_DIR = Path("outputs")
EXEC_DIR = Path("exec")

files_dict = FilesDict.from_file(OUTPUT_DIR)

ai = AI(model_name="gpt-4o")

env = ExecutionEnv(EXEC_DIR)

MAX_ITERATIONS = 10
for iteration in range(MAX_ITERATIONS):
    print(f"Iteration {iteration}")
    files_dict, exec_result = run_code(ai, env, files_dict)
    
    if exec_result.returncode == 0:
        break
    
    files_dict = fix_code(ai, files_dict, exec_result)
    
files_dict.to_file(OUTPUT_DIR)

print(files_dict)

