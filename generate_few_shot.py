# Generate the few_shot examples from pertinent papers

import os
import yaml
from tqdm import tqdm
from pathlib import Path

from utils.completion import post
from utils.extract import extract_json


import requests
import fitz  # PyMuPDF
import re
import argparse
import os

def download_pdf(pdf_url, output_filename):
    response = requests.get(pdf_url)
    with open(output_filename, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ''
        for page in doc:
            text += page.get_text()
    return text

def preprocess_text(text):
    # Remove newlines and extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def get_pdf_url_from_arxiv_link(arxiv_link):
    # Extract the arXiv ID from the URL (last part of the link)
    arxiv_id = arxiv_link.split('/')[-1]
    return f'https://arxiv.org/pdf/{arxiv_id}.pdf'

def process_arxiv_paper(arxiv_link):
    pdf_url = get_pdf_url_from_arxiv_link(arxiv_link)
    pdf_filename = f'{pdf_url.split("/")[-1]}'
    download_pdf(pdf_url, pdf_filename)
    extracted_text = extract_text_from_pdf(pdf_filename)
    clean_text = preprocess_text(extracted_text)
    os.remove(pdf_filename)
    return clean_text
    

SYSTEM_PROMPT = """You are an AI assistant that reviews research papers and provides assessments

Respond in the following format:

REVIEW:```review
<REVIEW>
```

ASSESSMENT JSON:
```json
<JSON>
```

In <REVIEW>, provide a concise review of the paper in plain text:
- Summarize the main research problem or hypothesis
- Outline the key methodologies and techniques used
- Highlight the main findings and their significance
- Discuss potential limitations or areas for improvement

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be objective and realistic in your ratings and assessments.
This JSON will be automatically parsed, so ensure the format is precise and that trailing commas are avoided."""

PAPER_REVIEW_PROMPT = """Review the following paper and provide an assessment of its feasibility and interestingness for potential implementation or further research.
{paper_text}"""

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, extract, and process a paper from arXiv.")
    parser.add_argument("arxiv_link", type=str, help="The full URL to the arXiv paper")
    args = parser.parse_args()
    
    text = process_arxiv_paper(args.arxiv_link)
    
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PAPER_REVIEW_PROMPT.format(paper_text=text)}
    ]
    review = post("openai/gpt-4o-2024-08-06", msg)
    
    review_json = extract_json(review)
    if review_json:
        filename = f"{review_json['Name']}.yaml"
        with open(filename, 'w') as file: yaml.dump(review_json, file, default_flow_style=False)
        print(f"Review saved to {filename}")
    else:
        print("Failed to extract JSON from the review.")