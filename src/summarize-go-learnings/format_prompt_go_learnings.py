import pandas as pd
import json
import sys
import os
import tiktoken
import logging


FORMAT_PROMPT_PRIMARY_PATH = "format_prompt.txt"
FORMAT_PROMPT_SECONDARY_PATH = "format_prompt_secondary.txt"
PROMPT_DATA_LENGTH_LIMIT = 5000
ENCODING_NAME = "cl100k_base"


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


def count_tokens(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def slice_dataframe(df, limit, encoding_name):
    df['count_temp'] = [count_tokens(x,encoding_name) for x in df['learning']]
    df['cumsum'] = df['count_temp'].cumsum()

    slice_index = None
    for i in range(1, len(df)):
        if df['cumsum'].iloc[i-1] <= limit and df['cumsum'].iloc[i] > limit:
            slice_index = i-1
            break

    if slice_index is not None:
        df_sliced = df.iloc[:slice_index+1]
                
    else:
        df_sliced = df    
    return df_sliced


def process_request_filter(request_filter):
    """Removes keys with empty values from the request filter."""
    request_filter = {k: v for k, v in request_filter.items() if v}
    return request_filter


def get_main_sectors(df):
    """Get only information from technical sectorial information"""
    temp = df[df['component']=='NS-specific areas of intervention']
    available_sectors= list(temp['sector'].unique())
    nb_sectors = len(available_sectors)
    if nb_sectors == 0:
        logging.info("There were not specific technical sectorial learnings")
        return []
    else:
        logging.info("Main sectors for secondary summaries selected")
        return available_sectors


def get_main_components(df):
    """Get information from components differents to the technical sectorial information"""
    temp = df[df['component']!='NS-specific areas of intervention']
    available_components= list(temp['component'].unique())
    nb_components= len(available_components)
    if nb_components == 0:
        logging.info("There were not specific components")
        return []
    else:
        logging.info("All components for secondary summaries selected")
        return available_components


def get_learnings_sector(sector, df):
    return df[df['sector']==sector]


def get_learnings_component(component, df):
    return df[df['component']==component]


def process_learnings_sector(sector,df, max_length_per_section):
    df = get_learnings_sector(sector, df).dropna()
    df_sliced = slice_dataframe(df, max_length_per_section, ENCODING_NAME)
    learnings_sector = '\n----------------\n'+"SUBTYPE: " + sector+'\n----------------\n'+'\n----------------\n'.join(df_sliced['learning'])
    return learnings_sector


def process_learnings_component(component,df, max_length_per_section):
    df = get_learnings_component(component, df).dropna()
    df_sliced = slice_dataframe(df, max_length_per_section, ENCODING_NAME)
    learnings_component = '\n----------------\n'+"SUBTYPE: " + component+'\n----------------\n'+'\n----------------\n'.join(df_sliced['learning'])
    return learnings_component


def process_data(type_prompt, df):
    """Process learnings from DataFrame according to type of summary"""
    if (type_prompt == "primary"):
        learnings_data = '\n----------------\n'.join(df['learning'].dropna())
        return learnings_data
    elif (type_prompt == "secondary"):
        sectors = get_main_sectors(df)
        components = get_main_components(df)
        max_length_per_section = PROMPT_DATA_LENGTH_LIMIT / (len(components)+len(sectors))
        learnings_sectors = '\n----------------\n\n'+"TYPE: SECTORS"+'\n----------------\n'.join([process_learnings_sector(x,df, max_length_per_section) for x in sectors if pd.notna(x)])
        learnings_components = '\n----------------\n\n'+"TYPE: COMPONENT"+'\n----------------\n'.join([process_learnings_component(x,df, max_length_per_section) for x in components if pd.notna(x)])
        learnings_data = learnings_sectors + learnings_components

        return learnings_data
    else:
        logging.error("Type of prompt is not valid. Type has to be either primary or secondary.")


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
        'in Emergency Response. You should prioritize the insights based on their recurrence and potential impact on humanitarian operations, and provide the top insights. \n\nI will pass you the DATA section, are you ready?\n\n'
    )

    return '\n'.join(instructions)


def build_data_section(type_prompt, df):
    """Builds the data section of the prompt from the DataFrame."""
    try:
        learnings_data = process_data(type_prompt,df)
        return f'DATA\n========================\n{learnings_data}\n\nI will pass you the FORMAT section, are you ready?\n\n'
    except Exception as e:
        logging.error(f"Error in generating summaries: {e}")
        raise

def get_format_section(type_prompt):
    """Reads the format section from a text file."""
    try:
        if (type_prompt == "primary"):
            content= read_file(FORMAT_PROMPT_PRIMARY_PATH)
            return content
        elif (type_prompt == "secondary"):
            content= read_file(FORMAT_PROMPT_SECONDARY_PATH)
            return content
        else:
            logging.error("Type of prompt is not valid. Type has to be either primary or secondary.")
    except Exception as e:
        logging.error(f"Error in generating summaries: {e}")
        raise


def create_prompt(prompt_intro, prompt_instruction, prompt_data, prompt_format):
    """Combines all sections to create the full prompt."""
    return ''.join([prompt_intro, prompt_instruction, prompt_data, prompt_format])


def format_prompt(request_filter_path, prioritized_learnings, type_prompt):
    """Formats the prompt based on request filter and prioritized learnings."""

    request_filter = read_json_file(request_filter_path)
    request_filter = process_request_filter(request_filter)

    prompt_intro = build_intro_section()
    prompt_instruction = build_instruction_section(request_filter, prioritized_learnings)
    prompt_data = build_data_section(type_prompt, prioritized_learnings)
    prompt_format = get_format_section(type_prompt)

    return create_prompt(prompt_intro, prompt_instruction, prompt_data, prompt_format)

def main(request_filter_path, prioritized_learnings, type_prompt):
    """Main function to generate the prompt."""
    try:
        prompt = format_prompt(request_filter_path, prioritized_learnings, type_prompt)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python format_prompt_go_learnings.py request_filter_path prioritized_learnings type_prompt")
    else:
        request_filter_path = sys.argv[1]
        prioritized_excerpts_path = sys.argv[2]
        type_prompt = sys.argv[3]
        prompt = main(request_filter_path, prioritized_learnings, type_prompt)
        print(prompt)
