import re
from typing import Literal

from ai_engineer.filesdict import FilesDict

LineType = Literal[" ", "-", "+"]
DiffType = Literal["new", "deleted", "same", "renamed"]


class Hunk:
    def __init__(self, start_line_pre:int, hunk_len_pre:int, start_line_post:int, hunk_len_post:int, content:str):
        self.start_line_pre, self.hunk_len_pre = start_line_pre-1, hunk_len_pre  # 0 indexed
        self.start_line_post, self.hunk_len_post = start_line_post-1, hunk_len_post
        self.content = content.lstrip("\n")

        self.lines = [line[1:] for line in self.content.split("\n")]

        self.types = [line[0] for line in self.content.split("\n")]

    @property
    def num_lines_changed(self) -> int:
        # this is very inaccurate and prone to LLM randomness
        return self.hunk_len_post - self.hunk_len_pre

    def increment_lines(self, increment:int):
        self.start_line_pre += increment
        self.start_line_post += increment

    def _verify_start_line(self, file_lines:list[str])->int:
        if self.types[0] in [" ", "-"]:
            real_idx = file_lines.index(self.lines[0])
        else:
            real_idx = max(0, self.start_line_pre-1)  # if first is add then we start one back to keep the logic the same

        if real_idx+1 != self.start_line_pre: print("Start line not same as specified")
        return real_idx

    def apply(self, file:str)->str:
        file_lines = file.split("\n")

        real_start_idx = self._verify_start_line(file_lines)

        for i, (line, line_type) in enumerate(zip(self.lines, self.types)):
            real_idx = real_start_idx + i
            if line_type == " ":
                assert file_lines[real_idx] == line, "Line not same as specified"
            elif line_type == "-":
                del file_lines[real_idx]
                real_start_idx -= 1
            elif line_type == "+":
                file_lines.insert(real_idx, line)
            else:
                raise ValueError(f"Invalid line type: {line_type}")
        
        return "\n".join(file_lines)
        
    def __str__(self):
        return f"@@ -{self.start_line_pre},{self.hunk_len_pre} +{self.start_line_post},{self.hunk_len_post} @@\n{self.content}"
    
    def __repr__(self):
        return f"Hunk(start_line_pre={self.start_line_pre}, hunk_len_pre={self.hunk_len_pre}, start_line_post={self.start_line_post}, hunk_len_post={self.hunk_len_post})"

def extract_all_hunks(diff:str)->list[Hunk]:
    hunk_pattern = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@([\s\S]*?)(?=\n@@|\Z)'
    matches = re.finditer(hunk_pattern, diff, re.MULTILINE)
    
    hunks = []
    for match in matches:
        hunk = Hunk(
            start_line_pre=int(match.group(1)),
            hunk_len_pre=int(match.group(2) or 1),  # when number of lines is not specified, it is 1
            start_line_post=int(match.group(3)),
            hunk_len_post=int(match.group(4) or 1),  # when number of lines is not specified, it is 1
            content=match.group(5)  # DO NOT STRIP
        )
        hunks.append(hunk)
    
    return hunks
    
class Diff:
    def __init__(self, pre_name:str, post_name:str, hunks:list[Hunk]):
        self.pre_name, self.post_name = pre_name, post_name
        self.hunks = sorted(hunks, key=lambda x: x.start_line_pre)

    @property
    def diff_type(self)->DiffType:
        same = self.pre_name == self.post_name
        if self.pre_name == "/dev/null" and not same: return "new"
        if self.post_name == "/dev/null" and not same: return "deleted"
        return "same" if same else "renamed"

    @classmethod
    def from_str(cls, diff:str):
        lines = diff.split("\n")

        first_header = lines[0].split(" ")
        second_header = lines[1].split(" ")
        assert len(first_header) == 2 and len(second_header) == 2, "Header should have two elements"

        first_pattern, first_file = first_header
        second_pattern, second_file = second_header
        assert first_pattern == "---" and second_pattern == "+++"

        hunks = extract_all_hunks("\n".join(lines[2:]))

        return cls(pre_name=first_file, post_name=second_file, hunks=hunks)

    def apply(self, files_dict:FilesDict)->FilesDict:
        if self.diff_type == "deleted":
            assert self.pre_name in files_dict, f"File {self.pre_name} not found"
            del files_dict[self.pre_name]
            return files_dict
        
        elif self.diff_type == "new":
            assert self.post_name not in files_dict, f"File {self.post_name} already exists"
            assert len(self.hunks) == 1, "New file cannot have more than one hunk"
            assert not any([t in [" ", "-"] for t in self.hunks[0].types]), "New file cannot have +'s or -'s"
            
            files_dict[self.post_name] = "\n".join(self.hunks[0].lines)
            return files_dict
        
        elif self.diff_type == "renamed":
            assert self.pre_name in files_dict, f"File {self.pre_name} not found"
            assert self.post_name not in files_dict, f"File {self.post_name} already exists"

            files_dict[self.post_name] = files_dict[self.pre_name]
            del files_dict[self.pre_name]
            self.diff_type = "same"

        # diff type is "same"
        content = files_dict[self.post_name]

        lines_changed = 0
        for hunk in self.hunks:
            hunk.increment_lines(lines_changed)
            content = hunk.apply(content)
            lines_changed += hunk.num_lines_changed
        
        print(f"Successfully applied diff to {self.post_name}")
        files_dict[self.post_name] = content  # sync to disk
        return files_dict
    
    def __str__(self):
        return f"--- {self.pre_name}\n+++ {self.post_name}\n" + "\n".join([str(hunk) for hunk in self.hunks])
    
    def __repr__(self):
        return f"Diff(pre_name={self.pre_name}, post_name={self.post_name}, diff_type={self.diff_type}, num_hunks={len(self.hunks)})"
        

def extract_all_diffs(text:str) -> list[Diff]:
    pattern = r"```diff\s*([\s\S]*?)\s*```"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    return [Diff.from_str(match.group(1).strip()) for match in matches]