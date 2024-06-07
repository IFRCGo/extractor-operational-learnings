import time
import sys
import logging
from query_go_learnings import query
from generate_prioritization_lists import generate_prioritization_list
from prioritize_components_go_learnings import prioritize_components
from contextualize_go_learnings import contextualize
from prioritize_excerpts_go_learnings import prioritize_excerpts
from format_prompt_go_learnings import format_prompt
from generate_summaries_go_learnings import generate_summaries


def summarize(request_filter_path, output_file_path):
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

        prioritized_learnings = prioritize_excerpts(contextualized_learnings)
        logging.info("Prioritized excerpts from learnings.")

        prompt = format_prompt(request_filter_path, prioritized_learnings)
        logging.info("Formatted the prompt.")

        generate_summaries(prompt, output_file_path)
        logging.info("Summarization process completed in %s seconds.", time.time() - start_time)
        
    except Exception as e:
        logging.error("An error occurred during the summarization process: %s", e)
        raise


def main(request_filter_path, output_file_path):
    summarize(request_filter_path, output_file_path)
    
  
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python summarize_go_learnings.py request_filter_path output_file_path")
    else:
        request_filter_path = sys.argv[1]
        output_file_path = sys.argv[2]
        main(request_filter_path, output_file_path)