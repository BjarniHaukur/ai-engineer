import re
from pathlib import Path

class FileDict(dict):
    @classmethod
    def from_response(cls, response:str):
        # Regex to match file paths and associated code blocks
        code_block_prefix_regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
        matches = re.finditer(code_block_prefix_regex, response, re.DOTALL)

        files_dict = {}
        for match in matches:
            # Clean and standardize the file path
            path = re.sub(r'[\:<>"|?*]', "", match.group(1))
            path = re.sub(r"^\[(.*)\]$", r"\1", path)
            path = re.sub(r"^`(.*)`$", r"\1", path)
            path = re.sub(r"[\]\:]$", "", path)

            # Extract and clean the code content
            content = match.group(2)

            files_dict[path.strip()] = content.strip()

        return cls(files_dict)
        
    @classmethod
    def from_file(cls, root_path:str|Path):
        root_path = Path(root_path)

        file_dict = {}
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(root_path)
                assert str(relative_path) not in file_dict, f"Duplicate file: {relative_path}"
                file_dict[str(relative_path)] = file_path.read_text()

        return cls(file_dict)

    
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

