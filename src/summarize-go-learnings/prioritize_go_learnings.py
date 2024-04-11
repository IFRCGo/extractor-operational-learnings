import pandas as pd
import requests
import json
import sys


def main(filtered_learnings_path, components_countries_path, components_regions_path, output_file_path):
    
    def fetch_url(field):       
        return requests.get(field).json()
    
    def fetch_field(field):
        
        dict_field = []
        temp_dict = requests.get('https://goadmin.ifrc.org/api/v2/'+field+'/?limit=200/').json()
        
        try:
            while temp_dict['next']:
                dict_field.extend(temp_dict['results'])
                temp_dict = fetch_url(temp_dict['next'])
            dict_field.extend(temp_dict['results'])
            
            return pd.DataFrame.from_dict(dict_field)
        
        except requests.exceptions.RequestException as e:
            # Handle HTTP request related exceptions
            print("HTTP request exception:", e)

        except pd.errors.ParserError as e:
            # Handle pandas parsing related exceptions
            print("Pandas parsing exception:", e)

        except Exception as e:
            # Catch any other unexpected exceptions
            print("Unexpected exception:", e)
    
    
    def add_country_region_id(df):
        df = pd.merge(df, go_country[['country_id','country_name']], on = 'country_name', how = 'left')
        df = pd.merge(df,go_region[['region_id','region_name']], on = 'region_name', how = 'left')
        
        return df
    
    
    def need_component_prioritization(df):
        '''From a filtered view of the GO ops learning dataset, verify if there is need to do prioritization or not'''
    
        nb_dif_components = len(df['component'].unique())
        nb_dif_learnings = len(df['learning'].unique())
    
        if (nb_dif_components > 3 and nb_dif_learnings > 3):
            return True
        else:
            return False
        
    def identify_type_prioritization(df):
        if len(df['country_id'].unique()) == 1:
            return 'single-country'
        elif len(df['region_id'].unique()) == 1:
            return 'single-region'
        elif len(df['region_id'].unique()) > 1:
            return 'multi-region'
        else:
            return None
    
    def add_new_component(prioritized_components, per_prioritized_components, df):
        available_components = list(df['component'].unique())
        available_components = [item for item in available_components if item not in prioritized_components]

        intersect_components = list(set(per_prioritized_components).intersection(set(available_components)))

        if len(intersect_components) > 0:
            #get components that are per prioritized and that are available in data (without counting already prioritized components)
            mask = [item in intersect_components for item in df['component']]
            temp = df[mask]
            component_counts = temp.value_counts('component')
            max_count = component_counts.max()
            most_frequent_components = list(component_counts[component_counts == max_count].index)
        else:
            #get components that are available in data (without counting already prioritized components)
            mask = [item in available_components for item in df['component']]
            temp = df[mask]
            component_counts = df.value_counts('component')
            max_count = component_counts.max()
            most_frequent_components = list(component_counts[component_counts == max_count].index)
        
        return prioritized_components + most_frequent_components
    
    def prioritize(df, type_prioritization = None):
    
        if (type_prioritization == 'single-country'):
            country_id = str(df['country_id'].iloc[0])
            per_prioritized_components = components_countries[country_id]
        
        elif (type_prioritization == 'single-region'):
            region_id = str(df['region_id'].iloc[0])
            per_prioritized_components = components_regions[region_id]
        
        else:
            per_prioritized_components = global_components
    
        component_counts = df.value_counts('component')
        max_count = component_counts.max()
        most_frequent_components = list(component_counts[component_counts == max_count].index)
        prioritized_components = most_frequent_components
    
        while len(prioritized_components) < 3:
            prioritized_components = add_new_component(prioritized_components, per_prioritized_components, df)
    
        mask = [x in prioritized_components for x in df['component']]
        return df[mask]
    
    def remove_country_region_id(df):
        df = df.drop(columns = ['country_id'], axis = 1)
        df = df.drop(columns = ['region_id'], axis = 1)
        return df
    
    
    go_country = fetch_field('country')
    go_country.rename(columns = {'id':'country_id','name':'country_name'}, inplace = True)

    go_region = fetch_field('region')
    go_region.rename(columns = {'id':'region_id'}, inplace = True)

    with open(components_countries_path) as json_file:
        components_countries = json.load(json_file)

    with open(components_regions_path) as json_file:
        components_regions = json.load(json_file)
        

    filtered_learnings = pd.read_csv(filtered_learnings_path)
    
    filtered_learnings = add_country_region_id(filtered_learnings)
    
    if need_component_prioritization(filtered_learnings):
        type_prioritization = identify_type_prioritization(filtered_learnings)
        prioritized_learnings = prioritize(filtered_learnings,type_prioritization)
    else:
        prioritized_learnings = filtered_learnings
        
    prioritized_learnings = remove_country_region_id(prioritized_learnings)
    prioritized_learnings.set_index('id', inplace = True)
    
    prioritized_learnings.to_csv(output_file_path, encoding = 'utf8')


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prioritize_go_learnings.py filtered_learnings_path components_countries_path components_regions_path output_file_path")
    else:
        filtered_learnings_path = sys.argv[1]
        components_countries_path = sys.argv[2]
        components_regions_path = sys.argv[3]
        output_file_path = sys.argv[4]
        main(filtered_learnings_path, components_countries_path, components_regions_path, output_file_path)