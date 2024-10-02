import re
from pathlib import Path

class FilesDict(dict):
    @classmethod
    def from_response(cls, response:str):
        # Regex to match file paths and associated code blocks
        code_block_prefix_regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
        matches = re.finditer(code_block_prefix_regex, response, re.DOTALL)

        files_dict = cls()
        for match in matches:
            # Clean and standardize the file path
            path = re.sub(r'[\:<>"|?*]', "", match.group(1))
            path = re.sub(r"^\[(.*)\]$", r"\1", path)
            path = re.sub(r"^`(.*)`$", r"\1", path)
            path = re.sub(r"[\]\:]$", "", path)

            # Extract and clean the code content
            content = match.group(2).strip()

            files_dict[path] = content 

        return files_dict
        
    @classmethod
    def from_file(cls, root_path:str|Path):
        root_path = Path(root_path)

        files_dict = cls()
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(root_path)
                assert str(relative_path) not in files_dict, f"Duplicate file: {relative_path}"
                files_dict[str(relative_path)] = file_path.read_text()

        return files_dict

    def to_context(self, enumerate_lines:bool=False):
        chat_str = ""
        for file_name, file_content in self.items():
            chat_str += f"{file_name}\n```"
            for i, file_line in enumerate(file_content.split("\n")):
                chat_str += f"{i+1} {file_line}\n" if enumerate_lines else f"{file_line}\n"
            chat_str += "```\n\n"
            
        return chat_str

    def to_file(self, root_path:str|Path):
        root_path = Path(root_path)
        root_path.mkdir(parents=True, exist_ok=True)

        for path, content in self.items():
            file_path = root_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)

    def __setitem__(self, key:str|Path, value:str): super().__setitem__(str(key), value)
    def __getitem__(self, key:str|Path)->str: return super().__getitem__(str(key))

