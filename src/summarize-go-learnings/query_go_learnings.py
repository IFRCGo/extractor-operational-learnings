import pandas as pd
import numpy as np
import requests
import json
import sys
import io
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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


def fetch_data_from_url(url):
    """Fetches data from a given URL and returns it as a DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Data fetched from URL: {url}")
        return pd.read_csv(io.StringIO(response.content.decode('utf8')))
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request exception: {e}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Pandas parsing exception: {e}")
        raise


def build_filtered_learning_url(request_filter, limit=200):
    """Constructs the API URL with given filters and limit."""
    base_url = 'https://goadmin.ifrc.org/api/v2/ops-learning/?'
    request_filter = {k: v for k, v in request_filter.items() if v}
    params = '&'.join([f"{k}={v}" for k, v in request_filter.items()])
    url = f"{base_url}{params}&limit={limit}"
    logging.info(f"URL built: {url}")
    return url
    
    
def fetch_filtered_learnings_csvexport(request_filter, limit=200):
    """Fetches filtered learning data and returns it as a pandas DataFrame."""
    url = build_filtered_learning_url(request_filter, limit)
    try:
        total_count = requests.get(url).json().get('count', 0)
        logging.info(f"Total records: {total_count}")

        dataframes = [fetch_data_from_url(f"{url}&format=csv&offset={i * limit}") 
                      for i in range((total_count // limit) + 1)]
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    except Exception as e:
        logging.error(f"Unexpected exception: {e}")
        raise


def query(request_filter_path):
    """Fetches the data based on the filter and writes it to a CSV file."""
    try:
        request_filter = read_json_file(request_filter_path)
        df = fetch_filtered_learnings_csvexport(request_filter)
        df.set_index('id', inplace=True)
        logging.info("Data successfully fetched and returned as DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Failed to query data: {e}")
        return pd.DataFrame()


def main(request_filter_path):
    """Main function to execute the query and return the DataFrame."""
    return query(request_filter_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python query_go_learnings.py request_filter_path")
    else:
        request_filter_path = sys.argv[1]
        df = main(request_filter_path)
        print(df)