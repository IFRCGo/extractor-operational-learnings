import time
import sys
import logging
import json
from query_go_learnings import query
from generate_prioritization_lists import generate_prioritization_list
from prioritize_components_go_learnings import prioritize_components
from contextualize_go_learnings import contextualize
from prioritize_excerpts_go_learnings import prioritize_excerpts
from format_prompt_go_learnings import format_prompt
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


def modify_summary(summary, type_summary):
    try:
        if type_summary == "primary":
            for key, value in summary.items():
                if key != "contradictory reports":
                    if isinstance(value, dict):
                        if "confidence level" not in value:
                            if value["content"].endswith("."):
                                value["confidence level"] = value["content"][-4:-1] 
                            else:
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
            #main bug found in responses
            for key, value in summary.items():
                if key != "contradictory reports":
                    if "confidence level" not in value:
                        logging.info("Summary doesn't explicitly state confidence level ")
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


def process_llm_summary(summary_file_path, type_summary, max_retries=3):
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
        
        retries += 1
        logging.warning(f"Retrying... Attempt {retries}/{max_retries}")
        time.sleep(1)  
    
    logging.error("Failed to get valid output from LLM after maximum retries.")


def summarize(request_filter_path, primary_output_file_path, secondary_output_file_path):
    """Summarizes the learnings based on the request filter."""
    start_time = time.time()
    
    try:
        logging.info("Starting the summarization process.")
        
        filtered_learnings = query(request_filter_path)
        logging.info("Queried and filtered learnings.")

        # Uncomment if needed for generating prioritization lists
        # generate_prioritization_list("../../../../../data/go/go_authorization_token.json", "list_components_countries.json", "list_components_regions.json", "list_components_global.json")
        
        prioritized_components_learnings = prioritize_components(
            filtered_learnings, 
            "list_components_countries.json", 
            "list_components_regions.json", 
            "list_components_global.json"
        )
        logging.info("Prioritized components learnings.")

        contextualized_learnings = contextualize(prioritized_components_learnings)
        logging.info("Contextualized the learnings.")

        prioritized_learnings = prioritize_excerpts(contextualized_learnings,"primary")
        logging.info("Prioritized excerpts from learnings.")
        primary_prompt = format_prompt(request_filter_path, prioritized_learnings,"primary")
        logging.info("Formatted the prompt for primary summary.")
        generate_summaries(primary_prompt, primary_output_file_path)
        logging.info("Generated the primary summary.")
        process_llm_summary(primary_output_file_path,"primary",3)

        prioritized_learnings = prioritize_excerpts(contextualized_learnings,"secondary")
        logging.info("Prioritized excerpts from learnings.")
        secondary_prompt = format_prompt(request_filter_path, prioritized_learnings,"secondary")
        logging.info("Formatted the prompt for secondary summary.")
        generate_summaries(secondary_prompt, secondary_output_file_path)
        logging.info("Generated the secondary summary.")
        process_llm_summary(secondary_output_file_path,"secondary",3)

        logging.info("Summarization process completed in %s seconds.", time.time() - start_time)
        
    except Exception as e:
        logging.error("An error occurred during the summarization process: %s", e)
        raise


def main(request_filter_path, primary_output_file_path, secondary_output_file_path):
    summarize(request_filter_path, primary_output_file_path, secondary_output_file_path)
    
  
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python summarize_go_learnings.py request_filter_path primary_output_file_path secondary_output_file_path")
    else:
        request_filter_path = sys.argv[1]
        primary_output_file_path = sys.argv[2]
        secondary_output_file_path = sys.argv[3]
        main(request_filter_path, primary_output_file_path, secondary_output_file_path)