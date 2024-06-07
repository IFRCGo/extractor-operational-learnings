import pandas as pd
import requests
import sys
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def contextualize(prioritized_learnings):
    def add_contextualization(df):
        """Adds appeal year and event name as a contextualization of the leannings."""
        for index, row in df.iterrows():
            df.at[index, 'learning'] = f"In {row['appeal_year']} in {row['appeal_name']}: {row['learning']}"

        df = df.drop(columns=['appeal_name'])
        logging.info("Contextualization added to DataFrame.")
        return df
    
    contextualized_learnings = add_contextualization(prioritized_learnings) 
    logging.info("Contextualization of learnings completed.")   
    return contextualized_learnings


def main(prioritized_learnings_path):
    return contextualize(prioritized_learnings_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python contextualize_go_learnings.py prioritized_learnings")
    else:
        prioritized_learnings = sys.argv[1]

        contextualized_learnings = main(prioritized_learnings)
        print(contextualized_learnings)