import os
import json
import logging
import tiktoken
import sys
from openai import AzureOpenAI
import ast


PROMPT_LENGTH_LIMIT = 7500
SYSTEM_MESSAGE_PATH = "system_message.txt"
ENCODING_NAME = "cl100k_base"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2023-05-15"
)


def save_as_json(data, output_file_path):
    """Saves data as a JSON file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error saving data to JSON file: {e}")
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


def validate_length_prompt(messages, prompt_length_limit):
    """Validates the length of the prompt."""
    message_content = [msg['content'] for msg in messages]
    text = ' '.join(message_content)
    count = count_tokens(text, ENCODING_NAME)
    logging.info(f"Token count: {count}")
    return count <= prompt_length_limit


def validate_text_is_dictionary(text):
    try:
        formatted_text = ast.literal_eval(text)
        return isinstance(formatted_text, dict)
    except: 
        return False 


def modify_format(summary):
    try:
        # Find the index of the last closing brace before the "Note"
        end_index = summary.rfind('}')

        # Truncate the string to include only the dictionary part
        formatted_summary = summary[:end_index+1]

        logging.info("Modification realized to response")
        return formatted_summary

    except Exception as e:
        logging.error(f"Modification failed: {e}")
        return "{}"


def validate_format(summary, output_file_path):
    try:
        # Attempt to parse the summary as a dictionary
        if validate_text_is_dictionary(summary):
            logging.info("Summary is a valid dictionary")
            save_as_json(ast.literal_eval(summary), output_file_path)
        else:
            formatted_summary = modify_format(summary)
            if validate_text_is_dictionary(formatted_summary):
                logging.info("Summary is a valid dictionary")
                save_as_json(ast.literal_eval(formatted_summary), output_file_path)
            else:
                logging.error("Summary is not a valid dictionary")
                save_as_json("{}", output_file_path)
    except (ValueError, SyntaxError) as e:
        # Log the error if the summary is not valid
        logging.error(f"Summary is not a valid dictionary: {e}")
        return None
    except Exception as e:
        # Catch all other exceptions
        logging.error(f"An unexpected error occurred: {e}")
        return None


def summarize(prompt, system_message = "You are a helpful assistant"):
    """Summarizes the prompt using the provided system message."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Understood, thank you for providing the data, and formatting requests. I am ready to proceed with the task."}
    ]
    
    if not validate_length_prompt(messages, PROMPT_LENGTH_LIMIT):
        logging.warning("The length of the prompt might be too long.")
        return "{}"

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            temperature=0
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        logging.error(f"Error in summarizing: {e}")
        raise


def generate_summaries(prompt, output_file_path):
    """Generates summaries using the provided system message and prompt."""
    try:
        system_message = read_file(SYSTEM_MESSAGE_PATH)
        summary = summarize(prompt,system_message)
        validate_format(summary, output_file_path)
        
    except Exception as e:
        logging.error(f"Error in generating summaries: {e}")
        raise


def main(prompt, output_file_path):
    """Main function to generate summaries."""
    return generate_summaries(prompt, output_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_summaries_go_learnings.py prompt output_file_path")
    else:
        prompt = sys.argv[1]
        output_file_path = sys.argv[2]
        summary = main(prompt,output_file_path)
        print(summary)
