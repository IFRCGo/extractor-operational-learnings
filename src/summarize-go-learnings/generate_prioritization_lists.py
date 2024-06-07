import pandas as pd
import numpy as np
import requests
import json
import sys
from ast import literal_eval
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


API_BASE_URL = 'https://goadmin.ifrc.org/api/v2/'


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


def export_as_json(data, output_file_path):
    """Exports a dictionary or DataFrame as JSON."""
    if isinstance(data, pd.DataFrame):
        data_dict = data.to_dict(orient='records')
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise ValueError("Input data must be a DataFrame or dictionary.")
    
    with open(output_file_path, "w") as outfile:
        json.dump(data_dict, outfile)
    logging.info(f"Data successfully exported to {output_file_path}")


def fetch_json_data(url, headers):
    """Fetches JSON data from a URL with optional headers."""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request exception: {e}")
        raise


def fetch_paginated_data(endpoint, headers, limit=200):
    """Fetches paginated data from a given API endpoint and returns a DataFrame."""
    url = f"{API_BASE_URL}{endpoint}/?limit={limit}"
    data_list = []
    
    while url:
        data_chunk = fetch_json_data(url, headers)
        data_list.extend(data_chunk.get('results', []))
        url = data_chunk.get('next')
        logging.info(f"Fetched {len(data_chunk.get('results', []))} records from {endpoint}")
    
    return pd.DataFrame(data_list)
    

def preprocess_country_data(df):
    """Preprocesses country data, filtering and renaming columns."""
    df_ns = df[df['society_name'].notna()]
    exclusion_list = ['IFRC Africa', '', 'IFRC Americas', 'IFRC Asia-Pacific', 'IFRC Europe', 'IFRC Geneva', 'IFRC MENA', 'Benelux ERU', 'ICRC']
    df_ns = df_ns[~df_ns['society_name'].isin(exclusion_list)]
    df_ns.rename(columns={'id': 'country'}, inplace=True)
    logging.info("Country data preprocessed.")
    return df_ns
 
    
def preprocess_per_overview_data(df):
    """Preprocesses PER overview data, extracting and renaming columns."""
    df['region'] = [x['region'] for x in df['country_details']]
    df['country'] = [x['id'] for x in df['country_details']]
    df.rename(columns ={'id':'overview'}, inplace = True)
    logging.info("PER overview data preprocessed.")    
    return df
    

def preprocess_prioritization_data(df, overview_df):
    """Preprocesses PER prioritization data, merging and filtering."""
    df = df[~df['is_draft']]
    df = df[df['prioritized_action_responses'].apply(lambda x: len(x) > 0)]
    df = df.merge(overview_df[['overview', 'country', 'region', 'assessment_number']], on='overview', how='left')
    df['components'] = df['prioritized_action_responses'].apply(lambda x: [item['component'] for item in x])
    #sort values to leave the latest assessments at the end, and keep only the latest
    df = df.sort_values('assessment_number').drop_duplicates(subset='country', keep='last')
    df = df[['region', 'country', 'components']]
    logging.info("PER prioritization data preprocessed.")
    return df

    
def generate_regional_prioritization_list(df):
    """Generates a list of regional prioritizations from the given data."""
    df_exploded = df.explode('components')
    regional_df = df_exploded.groupby(['region', 'components']).size().reset_index(name='count')
    # leave as prioritized regional components, those that are were prioritized at least in 3 countries
    regional_df = regional_df[regional_df['count'] > 2]
    regional_list = regional_df.groupby('region')['components'].apply(list).reset_index()
    logging.info("Regional prioritization list generated.")
    return regional_list


def generate_global_prioritization_list(regional_df):
    """Generates a global prioritization list from regional data."""
    global_df = regional_df.explode('components').groupby('components').size().reset_index(name='count')
    global_components = global_df[global_df['count'] > 2]['components'].tolist()
    global_list = {"global": global_components}
    logging.info("Global prioritization list generated.")
    return global_list


def generate_country_prioritization_list(regional_df, global_components, prioritization_df, country_df):
    """Generates a country-level prioritization list."""
    regional_dict = dict(zip(regional_df['region'], regional_df['components']))
    merged_df = country_df[['country', 'region']].merge(prioritization_df, on=['country', 'region'], how='left')
    no_prioritization_df = merged_df[merged_df['components'].isna()]
    
    for index, row in no_prioritization_df.iterrows():
        country_id = row['country']
        region_id = row['region']
        components = regional_dict.get(region_id, global_components['global'])
        no_prioritization_df.at[index, 'components'] = components
    
    final_df = pd.concat([merged_df.dropna(subset=['components']), no_prioritization_df])
    final_df['components'] = final_df['components'].apply(lambda x: literal_eval(str(x)))
    final_df = final_df[['country', 'components']]
    logging.info("Country prioritization list generated.")
    return final_df


def generate_prioritization_list(go_authorization_token_path,output_country_path, output_region_path, output_global_path):        
    """Generates and exports prioritization lists for country, regional, and global levels."""
    auth_token = read_json_file(go_authorization_token_path)
    headers = {'Authorization': auth_token['Authorization']}


    country_df = fetch_paginated_data('country', headers)
    per_overview_df = fetch_paginated_data('per-overview', headers)
    per_prioritization_df = fetch_paginated_data('public-per-prioritization', headers)


    country_df = preprocess_country_data(country_df)
    per_overview_df = preprocess_per_overview_data(per_overview_df)
    per_prioritization_df = preprocess_prioritization_data(per_prioritization_df, per_overview_df)

    
    regional_list = generate_regional_prioritization_list(per_prioritization_df)
    global_list = generate_global_prioritization_list(regional_list)
    country_list = generate_country_prioritization_list(regional_list, global_list, per_prioritization_df, country_df)
    
    export_as_json(regional_list, output_region_path)
    export_as_json(country_list, output_country_path)
    export_as_json(global_list, output_global_path)

    logging.info("Generation of prioritization lists completed.")

                                
def main(go_authorization_token_path,output_country_file_path, output_region_file_path, output_global_file_path):
    return generate_prioritization_list(go_authorization_token_path,output_country_file_path, output_region_file_path, output_global_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_prioritization_lists.py go_authorization_token_path output_country_file_path output_region_file_path output_global_file_path")
    else:
        go_auth_token_path = sys.argv[1]
        output_country_file_path = sys.argv[2]
        output_region_file_path = sys.argv[3]
        output_global_file_path = sys.argv[4]
        main(go_auth_token_path,output_country_file_path, output_region_file_path, output_global_file_path)

