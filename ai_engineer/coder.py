import subprocess
import os
import shutil

from ai_engineer.utils.completion import post
from ai_engineer.prompts import file_format_prompt, roadmap_prompt, philosophy_prompt, sandbox_prompt, fence, gen_bash_script_pre_prompt


def get_completion(prompt):
    return post("openai/gpt-4o", [{"role": "user", "content": prompt}])


implementation_prompt = '''

{roadmap_prompt}

{file_format_prompt}

{philosophy_prompt}

The following is the code you will write:

{fence}

{prompt}

{fence}

{sandbox_prompt}

'''

class File:
    def __init__(self, name:str, content:str):
        self.name = name
        self.content = content
    
    def __str__(self):
        return f"File({self.name}, len={len(self.content)})"
    
    def write(self, path:str = "."):
        import os
        # Create all necessary directories in the path
        full_path = os.path.join(path, self.name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(self.content)

import re
def find_files(text) -> list[File]:
    regex = r"```(\S*)\n(.+?)```"
    matches = re.findall(regex, text, re.DOTALL)
    files: list[File] = []
    for match in matches:
        file_name, content = match
        if file_name == "":
            continue
        files.append(File(file_name, content))
    return files


class Env:
    def __init__(self, name:str):
        # create a folder with the name
        os.makedirs(name, exist_ok=True)
        
        self.cwd = os.path.join(os.getcwd(), name)
        self.files: list[File] = []
        
    def set_files(self, files: list[File]):
        self.files = files
    
    def add_file(self, f: File):
        self.files.append(f)
    
    def write_files(self):
        for f in self.files:
            f.write(self.cwd)

    def run(self, commands: list[str], timeout: int = 7200):
        self.write_files()

        for command in commands:
                result = subprocess.run(
                    command.split(" "), 
                    cwd=self.cwd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, 
                    text=True, 
                    timeout=timeout
                )
                print(result.stdout)
                if result.stderr:
                    raise Exception(result.stderr)
                
    def cleanup(self):
        shutil.rmtree(self.cwd)
    
    def files_to_chat(self):
        return "\n".join([f"{f.name}" for f in self.files])
         
class Coder:
    def __init__(self):
        self.env = Env(name="env")
    
    def develop(self, prompt):
        self.implement(prompt)
        while True:
            try:
                self.run()
                break
            except Exception as e:
                self.fix(e)
        
    def implement(self, prompt):
        # given a prompt and files, generate code to achieve the prompt. Generate 
        prompt = implementation_prompt.format(
            roadmap_prompt=roadmap_prompt,
            file_format_prompt=file_format_prompt,
            philosophy_prompt=philosophy_prompt,
            fence=fence,
            prompt=prompt,
            sandbox_prompt=sandbox_prompt,
        )
        code = get_completion(prompt)
        
        files = find_files(code)
        self.env.set_files(files)
        return code

    def create_exec_script(self):
        
        prompt = gen_bash_script_pre_prompt + \
        "These are the files in the current directory: " + \
        self.env.files_to_chat() + \
        "Only respond with the code for the run.sh. Nothing else should be included."
        
        script = get_completion(prompt)
        script_file = File(name="run.sh", content=script)
        
        print(script_file.content)
        
        self.env.add_file(script_file)
        
        commands = script_file.content.split("\n") # TODO: parse the script file

        return commands
    
    def run(self):        
        # sets up an environment with dependencies and runs the code
        
        # alternatives:        
        # 1)    commands = self.create_exec_script()
        
        # 2)    add_exec_priv_command = "chmod +x run.sh"
        #       run_command = "./run.sh"
        
        
        commands = [
            "pip3 install -r requirements.txt", 
            "python3 src/main.py"
        ]
                
        self.env.run(commands = commands)


    def fix(self, error):
        pass
    
    def cleanup(self):
        self.env.cleanup()
        




if __name__ == "__main__":
    coder = Coder()
    coder.implement("Create a simple web server that listens on port 8000 and returns 'Hello, World!'")
