import re
from typing import Literal

from ai_engineer.filesdict import FilesDict

ActionType = Literal[" ", "-", "+"]
FileType = Literal["new", "deleted", "changed"]


class Hunk:
    def __init__(self, start_line_pre:int, hunk_len_pre:int, start_line_post:int, hunk_len_post:int, content:str):
        self.start_line_pre, self.hunk_len_pre = start_line_pre, hunk_len_pre
        self.start_line_post, self.hunk_len_post = start_line_post, hunk_len_post
        self.content = content

    @property
    def num_lines_changed(self) -> int:
        return self.hunk_len_post - self.hunk_len_pre

    def increment_lines(self, increment:int):
        self.start_line_pre += increment
        self.start_line_post += increment

    def apply(self, file:str)->str:
        # check whether content is valid (>90% similarity)

        pass
        

    def __str__(self):
        return f"@@ -{self.start_line_pre},{self.hunk_len_pre} +{self.start_line_post},{self.hunk_len_post} @@\n{self.content}"

def extract_all_hunks(diff:str)->list[Hunk]:
    hunk_pattern = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@([\s\S]*?)(?=\n@@|\Z)'
    matches = re.finditer(hunk_pattern, diff, re.MULTILINE)
    
    hunks = []
    for match in matches:
        hunk = Hunk(
            start_line_pre=int(match.group(1)),
            hunk_len_pre=int(match.group(2) or 1),
            start_line_post=int(match.group(3)),
            hunk_len_post=int(match.group(4) or 1),
            content=match.group(5).strip()
        )
        hunks.append(hunk)
    
    return hunks
    
class Diff:
    def __init__(self, file_name:str, file_type:FileType, hunks:list[Hunk]):
        self.file_name, self.file_type = file_name, file_type
        self.hunks = sorted(hunks, key=lambda x: x.start_line_pre)

    @classmethod
    def from_str(cls, diff:str):
        lines = diff.split("\n")

        first_header = lines[0].split(" ")
        second_header = lines[1].split(" ")
        assert len(first_header) == 2 and len(second_header) == 2

        first_pattern, first_file = first_header
        second_pattern, second_file = second_header
        assert first_pattern == "---" and second_pattern == "+++"

        same_file = first_file == second_file
        new_file = not same_file and first_file == "/dev/null"
        del_file = not same_file and second_file == "/dev/null"
        assert not new_file and not del_file, "Both files cannot be null"
        assert any([same_file, new_file, del_file]), "Could not read file name"

        hunks = extract_all_hunks(lines[2:])

        return cls(file_name=first_file, file_type="same" if same_file else "new" if new_file else "deleted", hunks=hunks)

    def apply(self, filesdict:FilesDict)->FilesDict:
        if self.file_type == "deleted":
            assert self.file_name in filesdict, f"File {self.file_name} not found"
            del filesdict[self.file_name]
            return filesdict
        
        elif self.file_type == "new":
            assert self.file_name not in filesdict, f"File {self.file_name} already exists"
            assert len(self.hunks) == 1 and self.hunks[0].num_lines_changed == 0, "New file cannot have hunks"
            assert not any([line[0] in ["+", "-"] for line in self.hunks[0].content.split("\n")]), "New file cannot have +'s or -'s"
            
            filesdict[self.file_name] = self.hunks[0].content  # strip hunk content of possible +'s 
            return filesdict

        content = filesdict[self.file_name]

        lines_changed = 0
        for hunk in self.hunks:
            hunk.increment_lines(lines_changed)
            content = hunk.apply(content)
            lines_changed += hunk.num_lines_changed

        filesdict[self.file_name] = content  # sync to disk
        return filesdict
        

def extract_all_diffs(text:str) -> list[Diff]:
    pattern = r"```diff\s*([\s\S]*?)\s*```"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    return [Diff.from_str(match.group(1).strip()) for match in matches]