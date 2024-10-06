import re
import shutil
from pathlib import Path

from ai_engineer.utils import is_inconsequential

class FilesDict(dict):
    """ A dictionary that maps file paths to corresponding code. Automatically syncs with files on disk. """
    def __init__(self, root_path:str|Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_path = Path(root_path)
        for path, content in self.items(): self[path] = content  # sync

    @classmethod
    def from_response(cls, response:str, root_path:str|Path):
        # Regex to match file paths and associated code blocks
        code_block_prefix_regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
        matches = re.finditer(code_block_prefix_regex, response, re.DOTALL)

        files_dict = cls(root_path)
        for match in matches:
            # Clean and standardize the file path
            path = re.sub(r'[\:<>"|?*]', "", match.group(1))
            path = re.sub(r"^\[(.*)\]$", r"\1", path)
            path = re.sub(r"^`(.*)`$", r"\1", path)
            path = re.sub(r"[\]\:]$", "", path)

            # Extract and clean the code content
            content = match.group(2).strip()

            files_dict[path] = content  # creates the file on disk

        return files_dict

    @classmethod
    def from_folder(cls, root_path:str|Path):
        files_dict = cls(root_path)
        for file_path in files_dict.root_path.rglob('*'):
            if is_inconsequential(file_path): continue  # skip things like .DS_Store

            if file_path.is_file():
                relative_path = file_path.relative_to(root_path)
                assert str(relative_path) not in files_dict, f"Duplicate file: {relative_path}"
                files_dict[str(relative_path)] = file_path.read_text()

        return files_dict
        
    def to_context(self, enumerate_lines:bool=False):
        chat_str = ""
        for file_name, file_content in self.items():
            chat_str += f"{file_name}\n```\n"
            for i, file_line in enumerate(file_content.split("\n")):
                chat_str += f"{i+1} {file_line}\n" if enumerate_lines else f"{file_line}\n"
            chat_str += "```\n\n"
            
        return chat_str

    def __contains__(self, key:str|Path)->bool:
        return super().__contains__(str(key))

    def __getitem__(self, key:str|Path)->str:
        return super().__getitem__(str(key))

    def __setitem__(self, key:str|Path, value:str):
        path = self.root_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        super().__setitem__(str(key), value)

    def __delitem__(self, key:str|Path):
        if key not in self: return
        path = self.root_path / key
        path.unlink()  # Delete file
        if path.parent.is_dir() and not path.parent.iterdir(): path.parent.rmdir()  # Remove empty parent directories
        super().__delitem__(str(key))

    def delete(self):  # can't do __del__ since we want the files to persist after the program ends.
        """Delete all files in the files_dict from the disk and deletes the directory if empty"""
        for path in list(self.keys()): del self[path]  # dictionary changes size while iterating
        
        if all(is_inconsequential(path) for path in self.root_path.rglob("*")):  # if no important files are left
            shutil.rmtree(self.root_path)
        else:
            raise Exception(f"Not deleting {self.root_path}. It still contains files not in the files_dict.")


