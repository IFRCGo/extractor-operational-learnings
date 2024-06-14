import pandas as pd
import tiktoken
import sys
from itertools import chain
import logging


PROMPT_DATA_LENGTH_LIMIT = 5000
ENCODING_NAME = "cl100k_base"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def remove_duplicates(df, type_prompt):
    """Remove duplicate rows based on the 'learning' and 'component' column."""
    if type_prompt == 'primary':
        return df.drop_duplicates(subset='learning')
    elif type_prompt == 'secondary':
        return df.drop_duplicates(subset=['learning','component'])
    else:
        logging.error('Type of prompt is not valid. Type has to be either primary or secondary.')
        return None


def count_tokens(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def sort_excerpts(df, type_prompt):
    """Sort DataFrame by 'appeal_year' in descending order."""
    if type_prompt == 'primary':
        df = df.sort_values(by='appeal_year', ascending=False)
        df.reset_index(inplace = True, drop = True)
        return df
    elif type_prompt == 'secondary':
        df_sorted = df.sort_values(by=['component', 'appeal_year'], ascending=[True, False])
        grouped = df_sorted.groupby('component')

        # Create an interleaved list of rows
        interleaved = list(chain(*zip(*[group[1].itertuples(index=False) for group in grouped])))

        # Convert the interleaved list of rows back to a DataFrame
        result = pd.DataFrame(interleaved)
        result.reset_index(inplace = True, drop = True)
        return result
    else:
        logging.error('Type of prompt is not valid. Type has to be either primary or secondary.')
        return None


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


def prioritize_most_recent(df, type_prompt, limit=2000, encoding_name="cl100k_base"):
    """Prioritize the most recent excerpts within the token limit."""
    df = remove_duplicates(df, type_prompt)
    df = sort_excerpts(df, type_prompt)
    return slice_dataframe(df, limit, encoding_name)
    

def prioritize_excerpts(contextualized_learnings, type_prompt):     
    """Main function to prioritize excerpts.""" 
    prioritized_excerpts_learnings = prioritize_most_recent(
        contextualized_learnings, 
        type_prompt,
        limit=PROMPT_DATA_LENGTH_LIMIT, 
        encoding_name=ENCODING_NAME,
    )
    logging.info("Prioritization of learnings completed.")
    return prioritized_excerpts_learnings


def main(contextualized_learnings, type_prompt):
    return prioritize_excerpts(contextualized_learnings, type_prompt)

  
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prioritize_excerpts_go_learnings.py contextualized_learnings type_prompt")
    else:
        contextualized_learnings = sys.argv[1]
        type_prompt = sys.argv[2]
        prioritized_learnings = main(contextualized_learnings, type_prompt)
        print(prioritized_learnings)