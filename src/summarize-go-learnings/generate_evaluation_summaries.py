import os
from openai import AzureOpenAI
import pandas as pd
import json
import logging
import tiktoken
import re
import numpy as np
import sys


ENCODING_NAME = "cl100k_base"


client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2023-05-15"
)


# Evaluation prompt template based on G-Eval - adapted from https://cookbook.openai.com/examples/evaluation/how_to_eval_abstractive_summarization
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""


evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}


def read_file(file_path):
    """Reads a file and returns its content as a string."""
    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            return f.read()
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise


def read_json_file(file_path):
    """Reads a JSON file and returns its content as a dictionary."""
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        logging.info("JSON file successfully read.")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        raise


def get_geval_score(
    criteria: str, steps: str, document: str, summary: str, metric_name: str
):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_EVALUATOR_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


def get_date(log_file):
    token_start = "INFO: Starting the summarization process on "
    token_end = "\nINFO: JSON file successfully read."

    start_index = log_file.find(token_start) + len(token_start)
    end_index = log_file.find(token_end, start_index)
        
    if start_index == -1 or end_index == -1:
        return "Tokens not found in the log file."
    
    date = log_file[start_index:end_index].strip()
    return date


def get_filters(log_file):
    token_start = "Summarize essential insights from the DATA in "
    token_end = "aspects in Emergency Response. \n"

    start_index = log_file.find(token_start) + len(token_start)
    end_index = log_file.find(token_end, start_index)
        
    if start_index == -1 or end_index == -1:
        return "Tokens not found in the log file."
    
    filters = log_file[start_index:end_index].strip()
    return filters


def get_document_summary(log_file, type_summary = 'primary'):
    token_start = "\nDATA\n========================\n"
    token_end = "\nI will pass you the FORMAT section, are you ready?\n"

    if type_summary == 'primary':
        start_index = log_file.find(token_start)
        end_index = log_file.find(token_end, start_index)
    elif type_summary == 'secondary':
        start_index = log_file.rfind(token_start)
        end_index = log_file.rfind(token_end, start_index)
        
    if start_index == -1 or end_index == -1:
        return "Tokens not found in the log file."
    
    document = log_file[start_index:end_index].strip()
    return document


def get_nb_excerpts_log(log_file, type_summary = 'primary'):
    if type_summary == 'primary':
        token_start = "INFO: Finalized processing primary summary."
        token_end = "INFO: Prioritization of learnings completed."
    elif type_summary == 'secondary':
        token_start = "INFO: Finalized processing secondary summary."
        token_end = "INFO: Complete summarization process done"

    start_index = log_file.find(token_start) + len(token_start)
    end_index = log_file.find(token_end, start_index)

    text = log_file[start_index:end_index].strip()
    numbers = re.findall(r'\d+', text)

    # Convert the numbers to integers
    figures = [int(num) for num in numbers]
    
    if start_index == -1 or end_index == -1:
        return "Tokens not found in the log file."

    
    nb_retrieved = figures[0]
    nb_prioritized = figures[1]
    return nb_retrieved, nb_prioritized


def get_nb_excerpts_displayed(summary, type_summary = 'primary'):
    excerpts_displayed = []
    if type_summary == 'primary':
        for key, value in summary.items():
                if key != "contradictory reports":
                    if isinstance(value, dict):
                        excerpts_displayed = excerpts_displayed + summary[key]["excerpts id"].split(',')
                        
    elif type_summary == 'secondary':
        for key, value in summary.items():
            excerpts_displayed = excerpts_displayed + summary[key]["excerpts id"].split(',')

    excerpts_displayed = np.unique([x.strip() for x in excerpts_displayed])
    nb_displayed = len(excerpts_displayed)
    return nb_displayed


def get_execution_time(log_file):
    token_start = "INFO: Complete summarization process done in"
    token_end = "seconds."
    start_index = log_file.find(token_start) + len(token_start)
    end_index = log_file.find(token_end, start_index)
        
    if start_index == -1 or end_index == -1:
        return "Tokens not found in the log file."
    
    text = log_file[start_index:end_index].strip()

    pattern = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    time = re.findall(pattern, text)[0][0]
    
    return time


def count_tokens(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def evaluate_summary(document, summary, type_summary = 'primary'):
    summaries = {type_summary : summary}
    data = {}

    for eval_type, (criteria, steps) in evaluation_metrics.items():
        for summ_type, content_summary in summaries.items():
            result = get_geval_score(criteria, steps, document, content_summary, eval_type)
            data[eval_type] = re.findall(r'\d+', result)[0]
    return data['Relevance'], data['Coherence'], data['Consistency'], data['Fluency']


def generate_evaluation(log_file_path, primary_summary_path, secondary_summary_path):
    """Generates evaluation using the provided log output and prompt."""
    try:
        log = read_file(log_file_path)

        date = get_date(log)

        filters = get_filters(log)

        primary_document = get_document_summary(log, type_summary = 'primary')
        secondary_document = get_document_summary(log, type_summary = 'secondary')

        primary_summary = read_json_file(primary_summary_path)
        secondary_summary = read_json_file(secondary_summary_path)

        nb_primary_retrieved, nb_primary_prioritized = get_nb_excerpts_log(log, type_summary = 'primary')
        nb_secondary_retrieved, nb_secondary_prioritized = get_nb_excerpts_log(log, type_summary = 'secondary')

        nb_primary_displayed= get_nb_excerpts_displayed(primary_summary)
        nb_secondary_displayed= get_nb_excerpts_displayed(secondary_summary)

        execution_time = get_execution_time(log)

        primary_input_tokens = count_tokens(str(primary_document),ENCODING_NAME)
        primary_output_tokens = count_tokens(str(primary_summary),ENCODING_NAME)

        secondary_input_tokens = count_tokens(str(secondary_document),ENCODING_NAME)
        secondary_output_tokens = count_tokens(str(secondary_summary),ENCODING_NAME)

        primary_relevance, primary_coherence, primary_consistency, primary_fluency = evaluate_summary(primary_document, primary_summary, "primary")

        secondary_relevance, secondary_coherence, secondary_consistency, secondary_fluency  = evaluate_summary(secondary_document, secondary_summary, "secondary")

        eval_dict = {"primary":{
                        "date":date,
                        "filters":filters,
                        "document":primary_document,
                        "summary":primary_summary,
                        "type":"primary",
                        "nb_retrieved":int(nb_primary_retrieved),
                        "nb_prioritized":int(nb_primary_prioritized),
                        "nb_displayed":int(nb_primary_displayed),
                        "execution_time":float(execution_time)/2,
                        "input_tokens":int(primary_input_tokens),
                        "output_tokens":int(primary_output_tokens),
                        "relevance":int(primary_relevance), 
                        "coherence": int(primary_coherence), 
                        "consistency":int(primary_consistency), 
                        "fluency":int(primary_fluency)
                        },
            'secondary':{"date":date,
                        "filters":filters,
                        "document":secondary_document,
                        "summary":secondary_summary,
                        "type":"secondary",
                        "nb_retrieved":int(nb_secondary_retrieved),
                        "nb_prioritized":int(nb_secondary_prioritized),
                        "nb_displayed":int(nb_secondary_displayed),
                        "execution_time":float(execution_time)/2,
                        "input_tokens":int(secondary_input_tokens),
                        "output_tokens":int(secondary_output_tokens),
                        "relevance":int(secondary_relevance), 
                        "coherence": int(secondary_coherence), 
                        "consistency":int(secondary_consistency), 
                        "fluency":int(secondary_fluency)
                        }
            }
        return eval_dict
    except Exception as e:
        logging.error(f"Error in evaluating summaries: {e}")
        raise


def main(log_file_path, primary_summary_path, secondary_summary_path):
    """Main function to generate summaries."""
    return generate_evaluation(log_file_path, primary_summary_path, secondary_summary_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_evaluation_go_learnings.py log_file_path primary_summary_path secondary_summary_path")
    else:
        log_file_path = sys.argv[1]
        primary_summary_path = sys.argv[2]
        secondary_summary_path = sys.argv[3]
        evaluation = main(log_file_path, primary_summary_path, secondary_summary_path)

