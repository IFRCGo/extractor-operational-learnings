import pandas as pd
import requests
import json
import sys

def main(prioritized_learnings_path, output_file_path):
    
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
       
    def add_contextualization(df):
        go_appeal = fetch_field('appeal')
        go_appeal.rename(columns = {'code':'appeal_code','name':'appeal_name'}, inplace = True)
        df = pd.merge(df, go_appeal[['appeal_code','appeal_name']], on = 'appeal_code', how = 'left')
        
        df['learning'] = ['In '+ str(x) + ' in ' + y + ': '+z for x,y,z in zip(df['appeal_year'],df['appeal_name'],df['learning'])]
        df = df.drop(columns = ['appeal_name'], axis = 1)
        return df
    
    
    
    prioritized_learnings = pd.read_csv(prioritized_learnings_path)
    
    contextualized_learnings = add_contextualization(prioritized_learnings)
    
    contextualized_learnings.set_index('id', inplace = True)
    
    contextualized_learnings.to_csv(output_file_path, encoding = 'utf8')
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python contextualize_go_learnings.py prioritized_learnings_path output_file_path")
    else:
        prioritized_learnings_path = sys.argv[1]
        output_file_path = sys.argv[2]

        main(prioritized_learnings_path, output_file_path)