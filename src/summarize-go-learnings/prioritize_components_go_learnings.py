import pandas as pd
import requests
import json
import sys
import logging


MIN_DIF_COMPONENTS = 3 
MIN_DIF_EXCERPTS = 3


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


def validate_df_not_empty(df):
    if df.empty:
        logging.info("Source dataframe is empty")
        return False
    else:
        return True
    
def need_component_prioritization(df):
    """Determines if prioritization is needed based on unique components and learnings."""
    nb_dif_components = len(df['component'].unique())
    nb_dif_learnings = len(df['learning'].unique())
    return nb_dif_components > MIN_DIF_COMPONENTS and nb_dif_learnings > MIN_DIF_EXCERPTS


def identify_type_prioritization(df):
    """Identifies the type of prioritization required based on the data."""
    if len(df['country_id'].unique()) == 1:
        return 'single-country'
    elif len(df['region_id'].unique()) == 1:
        return 'single-region'
    elif len(df['region_id'].unique()) > 1:
        return 'multi-region'
    else:
        return None
    

def add_new_component(prioritized_components, per_prioritized_components, df):
    """Adds new components to the prioritized list based on availability and frequency."""
    available_components = list(df['component'].unique())
    remaining_components = [item for item in available_components if item not in prioritized_components]

    intersect_components = list(set(per_prioritized_components) & set(remaining_components))

    if intersect_components:
        mask = df['component'].isin(intersect_components)
    else:
        mask = df['component'].isin(remaining_components)

    component_counts = df[mask]['component'].value_counts()
    most_frequent_components = component_counts[component_counts == component_counts.max()].index.tolist()
    
    return prioritized_components + most_frequent_components


def prioritize(df, components_countries, components_regions, components_global, type_prioritization=None):
    """Prioritizes components based on the type of prioritization."""
    if type_prioritization == 'single-country':
        country_id = str(df['country_id'].iloc[0])
        per_prioritized_components = components_countries.get(country_id, [])
    elif type_prioritization == 'single-region':
        region_id = str(df['region_id'].iloc[0])
        per_prioritized_components = components_regions.get(region_id, [])
    else:
        per_prioritized_components = components_global.get("global", [])

    component_counts = df['component'].value_counts()
    most_frequent_components = component_counts[component_counts == component_counts.max()].index.tolist()
    prioritized_components = most_frequent_components

    while len(prioritized_components) < 3:
        prioritized_components = add_new_component(prioritized_components, per_prioritized_components, df)
    
    mask = df['component'].isin(prioritized_components)
    return df[mask]


def prioritize_components(filtered_learnings, components_countries_path, components_regions_path, components_global_path):
    """Reads input files, prioritizes components, and returns the prioritized DataFrame."""
    
    if validate_df_not_empty(filtered_learnings):
        components_countries = read_json_file(components_countries_path)
        components_countries = {item['country']: item['components'] for item in components_countries}

        components_regions = read_json_file(components_regions_path)
        components_regions = {item['region']: item['components'] for item in components_regions}

        components_global = read_json_file(components_global_path)

        if need_component_prioritization(filtered_learnings):
            type_prioritization = identify_type_prioritization(filtered_learnings)
            prioritized_learnings = prioritize(filtered_learnings, components_countries, components_regions, components_global, type_prioritization)
        else:
            prioritized_learnings = filtered_learnings
        logging.info("Prioritization of components completed.")
        return prioritized_learnings
    else:
        return filtered_learnings


def main(filtered_learnings, components_countries_path, components_regions_path, components_global_path):

    return prioritize_components(filtered_learnings, components_countries_path, components_regions_path, components_global_path)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prioritize_components_go_learnings.py filtered_learnings components_countries_path components_regions_path components_global_path")
    else:
        filtered_learnings = sys.argv[1]
        components_countries_path = sys.argv[2]
        components_regions_path = sys.argv[3]
        components_global_path = sys.argv[4]
        prioritized_learnings = main(filtered_learnings, components_countries_path, components_regions_path, components_global_path)
        print(prioritized_learnings)