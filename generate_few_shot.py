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

In <JSON>, provide an assessment of the paper in JSON format with the following fields:
- "Title": The title of the paper being reviewed.
- "Authors": A list of the paper's authors.
- "Year": The year the paper was published.
- "Key_Contribution": A brief statement of the paper's main contribution.
- "Feasibility": A rating from 1 to 10 (lowest to highest) on how feasible it would be to implement or build upon this research.
- "Interestingness": A rating from 1 to 10 (lowest to highest) on how interesting or impactful this research is.
- "Potential_Extensions": A list of 2-3 potential ways to extend or build upon this research.

Be objective and realistic in your ratings and assessments.
This JSON will be automatically parsed, so ensure the format is precise and that trailing commas are avoided."""

PAPER_REVIEW_PROMPT = """Review the following paper and provide an assessment of its feasibility and interestingness for potential implementation or further research.
{paper_text}"""

# Main function to handle argparse and call the processing pipeline
def main():
    parser = argparse.ArgumentParser(description="Download, extract, and preprocess arXiv paper.")
    parser.add_argument("arxiv_link", type=str, help="The full URL to the arXiv paper")
    args = parser.parse_args()
    
    text = process_arxiv_paper(args.arxiv_link)
    
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PAPER_REVIEW_PROMPT.format(paper_text=text)}
    ]
    review = post("openai/gpt-4o-2024-08-06", msg)
    
    print(review)
    

    

if __name__ == "__main__":
    main()