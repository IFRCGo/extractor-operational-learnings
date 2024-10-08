import time
from datetime import datetime
import sys
import logging
from query_go_learnings import query
from generate_prioritization_lists import generate_prioritization_list
from prioritize_components_go_learnings import prioritize_components
from contextualize_go_learnings import contextualize
from prioritize_excerpts_go_learnings import prioritize_excerpts
from format_prompt_go_learnings import format_prompt
from generate_summaries_go_learnings import generate_summaries
from process_summaries_go_learnings import process_summary


def summarize(request_filter_path, primary_output_file_path, secondary_output_file_path):
    """Summarizes the learnings based on the request filter."""
    start_time = time.time()
    
    try:
        logging.info("Starting the summarization process.")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        logging.info("Starting the summarization process on %s.", dt_string)

        filtered_learnings = query(request_filter_path)
        logging.info("Queried and filtered learnings.")

        # Uncomment if needed for generating prioritization lists
        # generate_prioritization_list("../../../../../data/go/go_authorization_token.json", "list_components_countries.json", "list_components_regions.json", "list_components_global.json")
        # logging.info("Prioritized components lists generated.")

        contextualized_learnings = contextualize(filtered_learnings)
        logging.info("Contextualized the learnings.")

        prioritized_components_learnings = prioritize_components(
            contextualized_learnings, 
            "list_components_countries.json", 
            "list_components_regions.json", 
            "list_components_global.json"
        )
        logging.info("Prioritized components learnings.")        

        primary_prioritized_learnings = prioritize_excerpts(prioritized_components_learnings,"primary")
        logging.info("Prioritized excerpts from learnings for primary summary.")

        primary_prompt = format_prompt(request_filter_path, primary_prioritized_learnings,"primary")
        logging.info("Formatted the prompt for primary summary.")
        logging.info(primary_prompt)

        generate_summaries(primary_prompt, primary_output_file_path)
        logging.info("Generated the primary summary.")

        process_summary(primary_output_file_path,"primary", 3)
        logging.info("Finalized processing primary summary.")
        logging.info("%s learnings retrieved, %s learnings prioritized.", len(filtered_learnings),len(primary_prioritized_learnings))

        secondary_prioritized_learnings = prioritize_excerpts(contextualized_learnings,"secondary")
        logging.info("Prioritized excerpts from learnings for secondary summary.")

        secondary_prompt = format_prompt(request_filter_path, secondary_prioritized_learnings,"secondary")
        logging.info("Formatted the prompt for secondary summary.")
        logging.info(secondary_prompt)

        generate_summaries(secondary_prompt, secondary_output_file_path)
        logging.info("Generated the secondary summary.")

        process_summary(secondary_output_file_path,"secondary",3)
        logging.info("Finalized processing secondary summary.")
        logging.info("%s learnings retrieved, %s learnings prioritized.", len(filtered_learnings),len(secondary_prioritized_learnings))

        logging.info("Complete summarization process done in %s seconds.", time.time() - start_time)

        
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