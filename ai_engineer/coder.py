
from utils.completion import post
import subprocess

from prompts import file_format_prompt, roadmap_prompt, philosophy_prompt, sandbox_prompt, fence


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
        return f"File({self.name}, {self.content})"
    
    def write(self, path:str = "."):
        with open(path + "/" + self.name, "w") as f:
            f.write(self.content)

import re
def find_files(text) -> list[File]:
    regex = r"```\S*\n(.+?)```"
    matches = re.findall(regex, text, re.DOTALL)
    files: list[File] = []
    for match in matches:
        file_name = match.split("\n")[0]
        content = "\n".join(match.split("\n")[1:])
        files.append(File(file_name, content))
    return files


class Env:
    def __init__(self, cwd:str = "."):
        self.cwd = cwd
        self.files: list[File] = []
        
    def set_files(self, files: list[File]):
        self.files = files
    
    def write_files(self):
        for f in self.files:
            f.write(self.cwd)
    
    def run(self, commands: list[str], timeout: int = 7200):
        self.write_files()
        
        for command in commands:
            result = subprocess.run(
            command.split(" "), cwd=self.cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
         
class Coder:
    def __init__(self):
        self.env = Env()
    
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
        
    def run(self):
        # sets up an environment with dependencies and runs the code
        install_command = "pip install -r requirements.txt"
        run_command = "python main.py"
        
        self.env.run(commands = [install_command, run_command])


    def fix(self, error):
        # fix the code
        pass




if __name__ == "__main__":
    coder = Coder()
    coder.implement("Create a simple web server that listens on port 8000 and returns 'Hello, World!'", ["main.py", "requirements.txt"])
