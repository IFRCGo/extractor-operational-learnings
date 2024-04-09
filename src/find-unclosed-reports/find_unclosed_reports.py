import pandas as pd
import numpy as np
import requests
import json
import sys

def main(go_authorization_token_path, output_file_path):
    with open(go_authorization_token_path) as json_file:
        go_authorization_token = json.load(json_file)

        
    def fetch_url(field):
        return requests.get(field,headers = go_authorization_token).json()

    
    def fetch_field(field):
        dict_field = []
        temp_dict = requests.get('https://goadmin.ifrc.org/api/v2/'+field+'/?limit=200/', headers = go_authorization_token).json()
        try:
            while temp_dict['next']:
                dict_field.extend(temp_dict['results'])
                temp_dict = fetch_url(temp_dict['next'])
            dict_field.extend(temp_dict['results'])
            return pd.DataFrame.from_dict(dict_field)
        except:
            print('Problem accessing the table: ', field)
            print('========================')
            return None

        
    def fetch_unpublished_final_report(go_fr, ifrc_published_reports, go_region):
        '''get final reports that have not been published in GO'''
        go_fr_uc = go_fr[go_fr['is_published'] == False][['appeal_code', 'country_details', 'created_at', 'modified_at', 'is_published']]
        go_fr_uc['society_name'] = [x['society_name'] for x in go_fr_uc['country_details']]
        go_fr_uc['region'] = [x['region'] for x in go_fr_uc['country_details']]
     
        go_region.rename(columns = {'id':'region'}, inplace = True)
        go_fr_uc = pd.merge(go_fr_uc,go_region[['region','region_name']])
    
        go_fr_uc.drop(columns=['country_details','region'], inplace=True)
        appeal_code_list = list(go_fr_uc['appeal_code'])  
    
        '''get final reports that have been published in IFRC'''
        ifrc_published_reports['appeal_code'] = [x['code'] for x in ifrc_published_reports['appeal']]
        mask_1 = [x in appeal_code_list for x in ifrc_published_reports['appeal_code']]
        mask_2 = ['final' in x.lower() for x in (ifrc_published_reports['name'])]
        mask_3 = ['final' in x.lower() if pd.notna(x) else True for x in ifrc_published_reports['type']]
    
        mask = [(x and (y and z)) for x,y,z in zip(mask_1,mask_2,mask_3)]
        ifrc_final_published = ifrc_published_reports[mask]
    
        go_fr_uc = pd.merge(go_fr_uc,ifrc_final_published[['appeal_code','document_url']])
    
        return go_fr_uc

    
    def publish_findings(go_fr_uc, output_file_path):
        file_name = output_file_path+'unclosed_reports.xlsx'
        return go_fr_uc.to_excel(file_name, index = False)

    
    go_fr = fetch_field('dref-final-report')
    ifrc_published_reports = fetch_field('appeal_document')
    go_region = fetch_field('region')
    
    if go_fr is not None:
        unpublished_reports = fetch_unpublished_final_report(go_fr,ifrc_published_reports,go_region)
        publish_findings(unpublished_reports, output_file_path)

        if not unpublished_reports.empty:
            print('Reports published successfully!')
        else:
            print('No unpublished reports found.')
    else:
        print('No data fetched or an error occurred while fetching data.')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_unclosed_reports.py go_authorization_token_path output_path")
    else:
        go_authorization_token_path = sys.argv[1]
        output_file_path_arg = sys.argv[2]
        main(go_authorization_token_path, output_file_path_arg)