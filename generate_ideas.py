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
This JSON will be automatically parsed, so ensure the format is precise and that trailing commas are avoided.
"""

def generate_ideas(direction:str, num_ideas=3)->tuple[list[dict], list[str]]:
    assert direction in os.listdir(DIRECTIONS_PATH), f"Direction {direction} not found in {DIRECTIONS_PATH}"
       
    with open(DIRECTIONS_PATH / direction / "prompt.yaml", "r") as f: prompt = yaml.safe_load(f) or []
    with open(DIRECTIONS_PATH / direction / "ideas.yaml", "r") as f: prev_ideas = yaml.safe_load(f) or []
    with open(DIRECTIONS_PATH / direction / "template.py", "r") as f: code = f.read() or []

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
        
        if idea_description is None or idea_json is None:
            print(f"Failed to extract idea, continuing...")
            continue
        
        ideas.append(idea_json)
        thoughts.append(idea_description)
        
    return ideas, thoughts
    
def sort_by_score(ideas:list[dict], thoughts:list[str])->tuple[list[dict], list[str]]:
    scores = [idea["Interestingness"] + idea["Feasibility"] + idea["Novelty"] for idea in ideas]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ideas[i] for i in sorted_indices], [thoughts[i] for i in sorted_indices]
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, required=False, help="Specific research direction to generate ideas for")
    parser.add_argument("--num_ideas", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    
    if args.direction:
        assert args.direction in os.listdir(DIRECTIONS_PATH), f"Direction {args.direction} not found in {DIRECTIONS_PATH}"
        directions = [args.direction]
    else:
        directions = os.listdir(DIRECTIONS_PATH)
        
    
    ideas, thoughts = generate_ideas(directions[0], args.num_ideas)
    ideas, thoughts = sort_by_score(ideas, thoughts)
    ideas, thoughts = ideas[:args.top_k], thoughts[:args.top_k]
    
    # for idea, thought in zip(ideas, thoughts):
    #     # create the folders and save
        
        
    
    print(ideas)
    print("\n"*5)
    print(thoughts)

            
            
            # perhaps just return a list of ideas, to then be e.g. filtered before creating the folders
            
            # create the folder, save the idea (maybe _json is enough, otherwise we could ask for a more thorough writeup)
            
            # maybe do some reflection here, perhaps o1 does that well enough already
        
        