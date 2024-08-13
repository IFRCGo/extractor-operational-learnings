import json
import logging
import re
from generate_summaries_go_learnings import generate_summaries

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


def save_as_json(data, output_file_path):
    """Saves data as a JSON file."""
    try:
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data successfully saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error saving data to JSON file: {e}")
        raise
    
def validate_dict_not_empty(dct):
    if not bool(dct):
        logging.info("Source dict is empty")
        return False
    else:
        return True

def modify_summary(summary, type_summary):
    try:
        if type_summary == "primary":
            for key, value in summary.items():
                if key != "contradictory reports":
                    if isinstance(value, dict):
                        if "confidence level" not in value:
                            if value["content"].endswith("/5."):
                                value["confidence level"] = value["content"][-4:-1] 
                            elif(value["content"].endswith("/5")):
                                value["confidence level"] = value["content"][-3:] 
        return summary
    except Exception as e:
        logging.error(f"Modification failed: {e}")
        return summary


def validate_summary(summary, type_summary):
    try:        
        # Check if the structure matches the expected format
        if not isinstance(summary, dict):
            logging.info("Summary is not a dictionary")
            return False

        if (type_summary == "primary"):
            for key, value in summary.items():
                if key != "contradictory reports":
                    if "confidence level" not in value:
                        logging.info("Summary doesn't explicitly state confidence level ")
                        return False
                    if "excerpts id" not in value:
                        logging.info("Summary doesn't explicitly state excerpts")
                        return False
                    if not isinstance(value, dict):
                        logging.info("Each entry of the summaries doesn't have the expected structure")
                        return False
        elif(type_summary == "secondary"):
            for key, value in summary.items():
                if not isinstance(value, dict):
                    logging.info("First entry of summary is not a dictionary")
                    return False
                if "type" not in value or "subtype" not in value or "content" not in value:
                    logging.info("Each entry of the summaries doesn't have the expected structure")
                    return False
        logging.info("Validation of summary successful")         
        return True
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False
    

def process_summary(summary_file_path, type_summary, prompt, max_retries=3):
    summary = read_json_file(summary_file_path)

    if validate_dict_not_empty(summary):
        retries = 0
        while retries < max_retries:
            summary = read_json_file(summary_file_path)
            if validate_summary(summary, type_summary):
                return
            else:
                modified_summary = modify_summary(summary, type_summary)
                if validate_summary(modified_summary, type_summary):
                    save_as_json(modified_summary, summary_file_path)
                    return
                else:
                    generate_summaries(prompt, summary_file_path)
                    retries += 1
                    logging.warning(f"Retrying... Attempt {retries}/{max_retries}")
                    time.sleep(1)  
    
        logging.error("Failed to get valid output from LLM after maximum retries.")


def main(summary_file_path, type_summary, prompt):
    """Main function to process summaries."""
    return process_summary(summary_file_path, type_summary, prompt, max_retries=3)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_summaries_go_learnings.py summary_file_path type_summary prompt")
    else:
        summary_file_path = sys.argv[1]
        type_summary = sys.argv[2]
        prompt = sys.argv[3]
        main(summary_file_path,type_summary,prompt)