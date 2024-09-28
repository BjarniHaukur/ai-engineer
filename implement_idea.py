from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import os
import wandb
import subprocess
import shutil
import os.path as osp
import sys
from subprocess import TimeoutExpired

main_model = Model("gpt-4o")

MAX_RUNS = 3
MAX_ITERS = 10

MAX_STDERR_OUTPUT = 1500

main_prompt = """
Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

After you complete each change, we will run the command `python main.py'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS. WRITE THE TRAINING CODE INSIDE main.py

Only implement the code for the Model class.

Use the requirements.txt file to list all the packages required to run the script.

You can then implement the next thing on your list.
"""

def run_experiment(idea_id, run_num, research_direction_id, timeout=7200):
    folder_name = f"{research_direction_id}/{idea_id}"
    cwd = osp.abspath(folder_name)
    
    shutil.copy(
        osp.join(folder_name, "main.py"),
        osp.join(folder_name, f"run_{run_num}.py"),
    )

    # LAUNCH COMMAND
    command = [
        "python",
        "main.py",
        "--out_dir",
        f"run_{run_num}",
        "--wandb_project",
        "ai-engineer",
        "--wandb_name",
        f"{idea_id}/run_{run_num}",
        "--wandb_group",
        research_direction_id,
    ]
    try:
        result = subprocess.run(
            ["pip", "install", "-r", "requirements.txt"], cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
 
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "val_accuracy.txt"), "r") as f:
                val_accuracy = f.readline()
            results = f"Validation accuracy: {val_accuracy}"

            next_prompt = f"""Run {run_num} completed. Here are the results:
{results}

Decide if you want to try to improve the result.
We will then run the command `python main.py'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you think you can't improve the results, respond with 'ALL_COMPLETED'."""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt

def run_idea(title, idea, idea_name, research_direction_id):
    # Create the research_direction_id/idea_name folder if it doesn't exist
    idea_folder = osp.join(research_direction_id, f"{idea_name}")
    if not osp.exists(idea_folder):
        os.makedirs(idea_folder)

    # Copy logging.py to the idea folder
    src_logging_file = "template.py"
    dst_logging_file = osp.join(idea_folder, "main.py")
    shutil.copy(src_logging_file, dst_logging_file)

    io = InputOutput(
            yes=True, chat_history_file=f"{research_direction_id}/{idea_name}/{idea_name}_aider.txt"
        )
    
    main_file = osp.join(research_direction_id, f"{idea_name}", "main.py")
    requirements_file = osp.join(research_direction_id, f"{idea_name}", "requirements.txt")

    coder = Coder.create(
        main_model=main_model,
        fnames=[main_file, requirements_file],
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )

    next_prompt = main_prompt.format(title=title, idea=idea, max_runs=MAX_RUNS, research_direction_id=research_direction_id, idea_id=f"{idea_name}")

    run = 1
    current_iter = 0
    while run < MAX_RUNS + 1:
        coder_out = coder.run(next_prompt)
        # print(coder_out)
        if "ALL_COMPLETED" in coder_out:
            break
        return_code, next_prompt = run_experiment(f"{idea_name}", run, research_direction_id)
        
        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1

        if current_iter >= MAX_ITERS:
            current_iter = 0
            run += 1