
from utils.completion import post

def get_completion(prompt):
    return post("openai/gpt-4o", [{"role": "user", "content": prompt}])

file_listing_prompt = '''
Every *file listing* MUST use this format:
- First line: opening "```"
- Second line: the filename with any originally provided path
- ... entire content of the file ...
- Final line: closing "```"

e.g.

```main.py
print("Hello, world!")
```
'''

implementation_prompt = '''

Given the following files: {files}, generate code to achieve the prompt: {prompt}.

{file_listing_prompt}

Don't include any other text or example usages in your response.

'''

class Coder:
    def __init__(self):
        pass
    
    def develop(self, prompt, files):
        self.implement(prompt, files)
        while True:
            try:
                self.run()
                break
            except Exception as e:
                self.fix(e)

    def implement(self, prompt, files):
        # given a prompt and files, generate code to achieve the prompt. Generate 
        prompt = implementation_prompt.format(files=files, prompt=prompt, file_listing_prompt=file_listing_prompt)
        print(prompt)

        code = get_completion(prompt)

        return code

        
    def run(self):
        # init environment
        # install dependencies
        # run the code
        pass

    def fix(self, error):
        # fix the code
        pass

import re
def find_code_files(code):

    regex = r"```\S*\n(.+?)```"
    matches = re.findall(regex, code, re.DOTALL)
    for match in matches:
        file_name = match.split("\n")[0]
        code = "\n".join(match.split("\n")[1:])
        print(f"File Name: {file_name}")
        print(f"Code:\n{code}")


if __name__ == "__main__":
    coder = Coder()
    code = coder.implement("Create a simple web server that listens on port 8000 and returns 'Hello, World!'", ["main.py", "requirements.txt"])
    find_code_files(code)