import pandas as pd
import numpy as np
import requests
import json
import sys
from ast import literal_eval

def main(go_authorization_token_path,output_country_file_path, output_region_file_path):
    
    
    with open(go_authorization_token_path) as json_file:
        go_authorization_token = json.load(json_file)
    
    
    def fetch_url(field, headers = go_authorization_token):       
        return requests.get(field,headers = go_authorization_token).json()
    
    
    def fetch_field(field, headers = go_authorization_token):
        
        dict_field = []
        temp_dict = requests.get('https://goadmin.ifrc.org/api/v2/'+field+'/?limit=200/', headers = go_authorization_token).json()
        try:
            while temp_dict['next']:
                dict_field.extend(temp_dict['results'])
                temp_dict = fetch_url(temp_dict['next'], headers = go_authorization_token)
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
    
    
    def preprocess_country_data(go_country):
        
        go_ns = go_country[pd.notna(go_country['society_name'])]
        list_not_ns = [ 'IFRC Africa', '','IFRC Americas', 'IFRC Asia-Pacific', 'IFRC Europe', 'IFRC Geneva','IFRC MENA','Benelux ERU','ICRC']
        mask = [x not in list_not_ns for x in go_ns['society_name']]
        go_ns = go_ns[mask]
        go_ns.rename(columns ={'id':'country'}, inplace = True)
        
        return go_ns
    
    def preprocess_per_overview_data(go_per_overview):
        go_per_overview['region'] = [x['region'] for x in go_per_overview['country_details']]
        go_per_overview['country'] = [x['id'] for x in go_per_overview['country_details']]
        go_per_overview.rename(columns ={'id':'overview'}, inplace = True)
        
        return go_per_overview
    
    def preprocess_prioritization_data(go_per_prioritization, go_country, go_per_overview):
        go_per_prioritization = go_per_prioritization[go_per_prioritization['is_draft'] == False]
    
        mask = [len(x)>0 for x in go_per_prioritization['prioritized_action_responses']]
        go_per_prioritization = go_per_prioritization[mask]
    
    
        go_per_prioritization = pd.merge(go_per_prioritization,go_per_overview[['overview','country','region','assessment_number']], on = 'overview', how = 'left')
        go_per_prioritization['components'] = [[x['component'] for x in y] for y in go_per_prioritization['prioritized_action_responses']]
    
        #sort values to leave the latest assessments at the end
        go_per_prioritization = go_per_prioritization.sort_values('assessment_number')
    
        #and keep only latest values of assessment
        go_per_prioritization = go_per_prioritization.drop_duplicates(subset = 'country', keep='last')
    
        go_per_prioritization = go_per_prioritization[['region','country','components']]
        
        return go_per_prioritization
    
    def generate_regional_prioritization_list(go_per_prioritization):
        go_latest_ns_components = go_per_prioritization.explode('components')

        go_latest_region_components = go_latest_ns_components.groupby(['region','components'],as_index=False).count()

        # leave as prioritized regional components, those that are were prioritized at least in 3 countries
        mask = [x > 2 for x in go_latest_region_components['country']]
        go_latest_region_components = go_latest_region_components[mask]

        go_latest_region_components.rename(columns = {'country':'nb_countries'}, inplace = True)

        go_latest_region_components = go_latest_region_components[['region','components','nb_countries']]

        region_components = go_latest_region_components.groupby('region')['components'].agg(list).reset_index()

        return region_components
    
    def generate_country_prioritization_list(region_components, go_per_prioritization, go_country):
        dict_region_components = dict(zip(region_components['region'],region_components['components']))
        global_components = list(region_components.explode('components').groupby('components', as_index = False).count()['region'][region_components.explode('components').groupby('components', as_index = False).count()['region']>2].index)
        go_ns_prioritization = pd.merge(go_country[['country','region']],go_per_prioritization, how = 'left', on = ['country','region'])
    
        countries_with_prioritization = go_ns_prioritization[pd.notna(go_ns_prioritization['components'])]
        countries_to_prioritize = go_ns_prioritization[pd.isna(go_ns_prioritization['components'])]
        
        
        for i in range(0, len(countries_to_prioritize)):
            country_id = countries_to_prioritize['country'].iloc[i]
            region_id = countries_to_prioritize['region'].iloc[i]
        
            if (pd.notna(region_id)):
                components = dict_region_components[region_id] 
            else:
                components = global_components
        
            countries_to_prioritize.loc[countries_to_prioritize['country'] == country_id, 'components'] = str(components)
        
        countries_to_prioritize['components'] = [literal_eval(x) for x in countries_to_prioritize['components']]
    
        country_components = pd.concat([countries_with_prioritization, countries_to_prioritize], ignore_index=True)
        country_components = country_components[['country','components']]
            
        return country_components
    
    def export_as_json(df,output_file_path):
        dict_df =  dict(zip(df[df.columns[0]],df[df.columns[1]]))
        with open(output_file_path, "w") as outfile: 
            json.dump(dict_df, outfile)
        
        
    go_per_prioritization = fetch_field('public-per-prioritization', headers = go_authorization_token)
    
    go_per_overview = fetch_field('per-overview', headers = go_authorization_token)
    
    go_country = fetch_field('country', headers = go_authorization_token)
    
    go_country = preprocess_country_data(go_country)
    
    go_per_overview = preprocess_per_overview_data(go_per_overview)
    
    go_per_prioritization = preprocess_prioritization_data(go_per_prioritization, go_country, go_per_overview)
    
    region_components = generate_regional_prioritization_list(go_per_prioritization)
    
    country_components = generate_country_prioritization_list(region_components, go_per_prioritization, go_country)
    
    export_as_json(region_components, output_region_file_path)
    
    export_as_json(country_components, output_country_file_path)   
                                

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_prioritization_lists.py go_authorization_token_path output_country_file_path output_region_file_path")
    else:
        go_auth_token_path = sys.argv[1]
        output_country_file_path = sys.argv[2]
        output_region_file_path = sys.argv[3]
        main(go_auth_token_path,output_country_file_path, output_region_file_path)