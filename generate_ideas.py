import os
import json
from tqdm import tqdm   
from pathlib import Path

from utils.completion import post
from utils.extract import extract, extract_json

DIRECTIONS_PATH = Path("research_directions")

IDEA_PROMPT = """{task_description}
{data_description}

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:```thought
<THOUGHT>
```

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, provide a high-level technical overview of the proposed research in plain text:
- Formulate the abstract research problem or hypothesis
- Specify the theoretical foundations, including relevant mathematical formulations
- Detail the proposed model architecture, including: Layer compositions and interactions or novel architectural components or modifications
- Outline the training process, including: Loss function formulation and justification, optimization techniques and learning rate strategies and any proposed regularization methods


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
       
    with open(DIRECTIONS_PATH / direction / "prompt.json", "r") as f: prompt = json.load(f)
    with open(DIRECTIONS_PATH / direction / "few_shot_ideas.json", "r") as f: few_shot_ideas = json.load(f)
    ideas, thoughts = [], []
    
    for _ in range(num_ideas):
        idea_prompt = IDEA_PROMPT.format(
            task_description=prompt["task_description"],
            data_description=prompt["data_description"],
            prev_ideas_string=json.dumps(few_shot_ideas) + json.dumps(ideas),
        )
        msg = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": idea_prompt}
        ]
        
        idea = post("openai/o1-preview-2024-09-12", msg)
        # idea = post("openai/gpt-4o-2024-08-06", msg)

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
    from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    all_ideas = {}
    all_thoughts = {}
    
    with ProcessPoolExecutor() as executor:
        future_to_direction = {executor.submit(generate_ideas, direction, args.num_ideas): direction for direction in directions}
        for future in tqdm(as_completed(future_to_direction), total=len(directions)):
            direction = future_to_direction[future]
            try:
                ideas, thoughts = future.result()
                ideas, thoughts = sort_by_score(ideas, thoughts)
                ideas, thoughts = ideas[:args.top_k], thoughts[:args.top_k]
                all_ideas[direction] = ideas
                all_thoughts[direction] = thoughts
                print(f"Completed processing for direction: {direction}")
            except Exception as exc:
                print(f"Direction {direction} generated an exception: {exc}")


    for direction in directions:            
        direction_path = DIRECTIONS_PATH / direction
        
        with open(direction_path / "ideas.json", "a") as f: json.dump(all_ideas[direction], f, indent=2)
        
        for idea, thought in zip(all_ideas[direction], all_thoughts[direction]):
            idea_path = direction_path / idea["Name"]
            idea_path.mkdir(parents=True, exist_ok=True)
            
            with open(idea_path / "thought.txt", "w") as f: f.write(thought)
