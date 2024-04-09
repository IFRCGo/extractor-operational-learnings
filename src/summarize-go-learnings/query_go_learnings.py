import pandas as pd
import numpy as np
import requests
import json
import sys
import io


def main(go_authorization_token_path, output_file_path, request_filter_path):
    
    '''
    The query go learnings function orchestrates the process of fetching filtered learning data from the GoAdmin API 
    and exporting it to a CSV file. It accepts paths to authorization token, output file, and request filter as input parameters.
    
    Parameters:
    - go_authorization_token_path: Path to the JSON file containing the GoAdmin authorization token.
    - output_file_path: Path where the fetched data will be exported as a CSV file.
    - request_filter_path: Path to the JSON file containing the request filter parameters.

    The request filter parameters are assumed to be stored in a dictionary format, 
    where keys represent different filtering criteria and values represent corresponding filter values.
    The function expects certain keys to be present in the request filter dictionary.
    The filtering criteria are applied to fetch a filtered version of the learning data from the GoAdmin API.
    '''
    
    with open(go_authorization_token_path) as json_file:
        go_authorization_token = json.load(json_file)
    
    with open(request_filter_path) as json_file:
        request_filter = json.load(json_file)
    
    def build_filtered_learning_url(request_filter):
        url_base = 'https://goadmin.ifrc.org/api/v2/ops-learning/?'
    
        empty_keys = [key for key, value in request_filter.items() if not value]
        for key in empty_keys:
            del request_filter[key]
    
        for arg in request_filter:
            url_base = url_base + arg + '=' + request_filter[arg] + '&'

        url = url_base + 'limit=200'
        return url
    
    
    def fetch_filtered_learnings_csvexport(request_filter, go_authorization_token):
        url = build_filtered_learning_url(request_filter)
        try:
            length = requests.get(url, headers = go_authorization_token).json()['count']
            r = requests.get(url+'&format=csv', headers = go_authorization_token)
            go_field = pd.read_csv(io.StringIO(r.content.decode('utf8')))

            for i in range(1,int(np.ceil(length/200))):
                start_offset = i*limit
                r = requests.get(url+'&format=csv'+'&offset='+str(start_offset), headers = go_authorization_token)
                go_field = pd.concat([go_field, pd.read_csv(io.StringIO(r.content.decode('utf8')))])
            return go_field
    
        except requests.exceptions.RequestException as e:
            # Handle HTTP request related exceptions
            print("HTTP request exception:", e)

        except pd.errors.ParserError as e:
            # Handle pandas parsing related exceptions
            print("Pandas parsing exception:", e)

        except Exception as e:
            # Catch any other unexpected exceptions
            print("Unexpected exception:", e)
            
    df = fetch_filtered_learnings_csvexport(request_filter,go_authorization_token)
    
    if not df.empty:
        df.to_csv(output_file_path,encoding = 'utf-8')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python query_go_learnings.py go_authorization_token_path output_path request_filter_path")
    else:
        go_authorization_token_path = sys.argv[1]
        output_file_path_arg = sys.argv[2]
        request_filter_path = sys.argv[3]
        main(go_authorization_token_path, output_file_path_arg,request_filter_path)