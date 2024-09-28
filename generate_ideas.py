import os
import yaml
from tqdm import tqdm   
from pathlib import Path

from utils.completion import post
from utils.extract import extract, extract_json

DIRECTIONS_PATH = Path("research_directions")

IDEA_PROMPT = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
```thought
<THOUGHT>
```

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, provide a high-level technical overview of the proposed research in plain text:
- Formulate the abstract research problem or hypothesis
- Conceptualize the methodological framework and experimental paradigm
- Identify key algorithms, models, or techniques to be utilized

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
"""

def generate_ideas(direction:str|None=None, num_ideas=3)->tuple[list[dict], list[str]]:
    if direction:
        assert direction in os.listdir(DIRECTIONS_PATH), f"Direction {direction} not found in {DIRECTIONS_PATH}"
        direction_paths = [DIRECTIONS_PATH / direction]
    else:
        direction_paths = [d for d in DIRECTIONS_PATH.iterdir() if d.is_dir()]
        
    
    for direction_path in direction_paths:        
        with open(direction_path / "prompt.yaml", "r") as f: prompt = yaml.safe_load(f) or []
        with open(direction_path / "ideas.yaml", "r") as f: prev_ideas = yaml.safe_load(f) or []
        with open(direction_path / "template.py", "r") as f: code = f.read() or []

        ideas, thoughts = [], []
        
        for _ in tqdm(range(num_ideas)):
            idea_prompt = IDEA_PROMPT.format(
                task_description=prompt["task_description"],
                code=code,
                prev_ideas_string=yaml.dump(prev_ideas),
            )
            msg = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": idea_prompt}
            ]
            
            # idea = post("openai/o1-preview-2024-09-12", msg)
            idea = post("openai/gpt-4o-2024-08-06", msg)

            idea_description = extract(idea, "thought")
            idea_json = extract_json(idea)
            
            ideas.append(idea_json)
            thoughts.append(idea_description)
            
        return ideas, thoughts
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, required=False, help="Specific research direction to generate ideas for")
    parser.add_argument("--num_ideas", type=int, default=5)
    args = parser.parse_args()
    
    ideas, thoughts = generate_ideas(args.direction, args.num_ideas)
    
    print(ideas)
    print("\n"*5)
    print(thoughts)

            
            
            # perhaps just return a list of ideas, to then be e.g. filtered before creating the folders
            
            # create the folder, save the idea (maybe _json is enough, otherwise we could ask for a more thorough writeup)
            
            # maybe do some reflection here, perhaps o1 does that well enough already
        
        