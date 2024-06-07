import pandas as pd
import tiktoken
import sys
import logging


PROMPT_DATA_LENGTH_LIMIT = 5000
ENCODING_NAME = "cl100k_base"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def remove_duplicates(df):
    """Remove duplicate rows based on the 'learning' column."""
    return df.drop_duplicates(subset='learning')


def count_tokens(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def sort_excerpts(df):
    """Sort DataFrame by 'appeal_year' in descending order."""
    return df.sort_values(by='appeal_year', ascending=False)


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


def prioritize_most_recent(df, limit=2000, encoding_name="cl100k_base"):
    """Prioritize the most recent excerpts within the token limit."""
    df = remove_duplicates(df)
    df = sort_excerpts(df)
    return slice_dataframe(df, limit, encoding_name)
    

def prioritize_excerpts(contextualized_learnings):     
    """Main function to prioritize excerpts.""" 
    prioritized_excerpts_learnings = prioritize_most_recent(
        contextualized_learnings, 
        limit=PROMPT_DATA_LENGTH_LIMIT, 
        encoding_name=ENCODING_NAME
    )
    logging.info("Prioritization of learnings completed.")
    return prioritized_excerpts_learnings


def main(contextualized_learnings):
    return prioritize_excerpts(contextualized_learnings)

  
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prioritize_excerpts_go_learnings.py contextualized_learnings")
    else:
        contextualized_learnings = sys.argv[1]
        prioritized_learnings = main(contextualized_learnings)
        print(prioritized_learnings)