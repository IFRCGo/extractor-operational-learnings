import pandas as pd
import json
import sys
import os
import logging


FORMAT_PROMPT_PATH = "format_prompt.txt"


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


def read_file(file_path):
    """Reads a file and returns its content as a string."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise


def process_request_filter(request_filter):
    """Removes keys with empty values from the request filter."""
    request_filter = {k: v for k, v in request_filter.items() if v}
    return request_filter


def build_intro_section():
    """Builds the introductory section of the prompt."""
    return "I will provide you with a set of instructions, data, and formatting requests in three sections. I will pass you the INSTRUCTIONS section, are you ready?"+ os.linesep + os.linesep


def build_instruction_section(request_filter, df):
    """Builds the instruction section of the prompt based on the request filter and DataFrame."""
    instructions = ['INSTRUCTIONS', '========================', 'Summarize essential insights from the DATA']

    if 'appeal_code__dtype__in' in request_filter:
        dtypes = df['dtype_name'].dropna().unique()
        dtype_str = '", "'.join(dtypes)
        instructions.append(f'concerning "{dtype_str}" occurrences')

    if 'appeal_code__country__in' in request_filter:
        countries = df['country_name'].dropna().unique()
        country_str = '", "'.join(countries)
        instructions.append(f'in "{country_str}"')

    if 'appeal_code__region' in request_filter:
        regions = df['region_name'].dropna().unique()
        region_str = '", "'.join(regions)
        instructions.append(f'in "{region_str}"')

    if 'sector_validated__in' in request_filter:
        sectors = df['sector'].dropna().unique()
        sector_str = '", "'.join(sectors)
        instructions.append(f'focusing on "{sector_str}" aspects')

    if 'per_component_validated__in' in request_filter:
        components = df['component'].dropna().unique()
        component_str = '", "'.join(components)
        instructions.append(f'and "{component_str}" aspects')

    instructions.append(
        'in Emergency Response. You should prioritize the insights based on their recurrence and potential impact on humanitarian operations, and provide the top 3 insights. Describe step by step your thought process.\n\nI will pass you the DATA section, are you ready?\n\n'
    )

    return '\n'.join(instructions)


def build_data_section(df):
    """Builds the data section of the prompt from the DataFrame."""
    learnings_data = '\n----------------\n'.join(df['learning'].dropna())
    return f'DATA\n========================\n{learnings_data}\n\nI will pass you the FORMAT section, are you ready?\n\n'


def get_format_section():
    """Reads the format section from a text file."""
    try:
        content= read_file(FORMAT_PROMPT_PATH)
        return content
    except Exception as e:
        logging.error(f"Error in generating summaries: {e}")
        raise


def create_prompt(prompt_intro, prompt_instruction, prompt_data, prompt_format):
    """Combines all sections to create the full prompt."""
    return ''.join([prompt_intro, prompt_instruction, prompt_data, prompt_format])


def format_prompt(request_filter_path, prioritized_learnings):
    """Formats the prompt based on request filter and prioritized learnings."""
    request_filter = read_json_file(request_filter_path)
    request_filter = process_request_filter(request_filter)

    prompt_intro = build_intro_section()
    prompt_instruction = build_instruction_section(request_filter, prioritized_learnings)
    prompt_data = build_data_section(prioritized_learnings)
    prompt_format = get_format_section()

    return create_prompt(prompt_intro, prompt_instruction, prompt_data, prompt_format)

def main(request_filter_path, prioritized_learnings):
    """Main function to generate the prompt."""
    try:
        prompt = format_prompt(request_filter_path, prioritized_learnings)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python format_prompt_go_learnings.py request_filter_path prioritized_learnings")
    else:
        request_filter_path = sys.argv[1]
        prioritized_excerpts_path = sys.argv[2]
        prompt = main(request_filter_path, prioritized_learnings)
        print(prompt)