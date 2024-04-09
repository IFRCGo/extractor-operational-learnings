import pandas as pd
import numpy as np
import requests
import json
import retrying
from retrying import retry
import time
from nltk.tokenize import LineTokenizer
import logging

CLASSIFY_URL = "https://dreftagging.azurewebsites.net/classify"
GO_API_URL = "https://goadmin.ifrc.org/api/v2/"
OPS_LEARNING_URL = GO_API_URL + "ops-learning/"
LIMIT_200 = "/?limit=200/"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def fetch_data(dref_final_report, appeal, ops_learning, go_auth_token_path):
    
    with open(go_auth_token_path) as json_file:
        go_authorization_token = json.load(json_file)

    def fetchUrl(field):
        return requests.get(field,headers = go_authorization_token).json()

    def fetchField(field):
        dict_field = []
        temp_dict = requests.get(GO_API_URL+field+LIMIT_200, headers = go_authorization_token).json()
        while temp_dict['next']:
            dict_field.extend(temp_dict['results'])
            temp_dict = fetchUrl(temp_dict['next'])
        dict_field.extend(temp_dict['results'])
        return pd.DataFrame.from_dict(dict_field)

    #read dref final reports, to extract learnings in planned interventions
    logging.info('Fetching DREF Final Reports from GO')
    dref_final_report = fetchField(dref_final_report)

    #read appeals to verify which drefs (appeals) are public and which drefs (appeals) are silent
    logging.info('Fetching Appeals from GO')
    appeals = fetchField(appeal)

    #read ops learning to verify which drefs have already been processed
    logging.info('Fetching Operational Learnings from GO')
    ops_learning = fetchField(ops_learning)
    ops_learning['appeal_code'] = [x['code'] for x in ops_learning['appeal']]
    
    return dref_final_report, appeals, ops_learning


def filter_final_report(final_report, appeal, ops_learning, final_report_is_published = True, appeal_is_published = True, in_ops_learning = False):
    
    if final_report_is_published:
        logging.info('Filtering only DREF Final Reports that have been closed')
        mask = [x for x in final_report['is_published']]
        final_report = final_report[mask]
        
    if appeal_is_published:
        logging.info('Filtering only DREF Final Reports that are public')
        mask = [x in list(appeal['code']) for x in final_report['appeal_code']]
        final_report = final_report[mask]
        
        
    if not in_ops_learning:
        logging.info('Filtering only DREF Final Reports that have not been processed yet for operational learning')
        list_new_reports = np.setdiff1d(final_report['appeal_code'].unique(),ops_learning['appeal_code'].unique())
        
        #only reports that are not processed yet
        mask = [x in list_new_reports for x in final_report['appeal_code']]
        final_report = final_report[mask]
        
    if final_report.empty:
        logging.warning('There were not find any DREF Final Reports after the filtering process')
        return None
    
    else:
        filtered_final_report = final_report[['appeal_code','planned_interventions']]
        logging.info('There were found %s reports after the filtering process', str(len(filtered_final_report)))
        return filtered_final_report


def split_rows(filtered_final_report):
    
    def split_planned_interventions(df):
        logging.info('Splitting DREF Final Reports per planned intervention')
        df = df.explode(column = 'planned_interventions', ignore_index = True)
    
        df['Sector'] = [x['title_display'] for x in df['planned_interventions']]
        df['Lessons Learnt'] = [x['lessons_learnt'] for x in df['planned_interventions']]
        df['Challenges'] = [x['challenges'] for x in df['planned_interventions']]
    
        mask_1 = [pd.notna(x) for x in df['Lessons Learnt']]
        mask_2 = [pd.notna(x) for x in df['Challenges']]
        mask = [x or y for x,y in zip(mask_1,mask_2)]
        df = df[mask]
        df.drop(columns = 'planned_interventions', inplace =True)
        
        df = df.melt(id_vars = ['appeal_code','Sector'],value_vars = ['Lessons Learnt','Challenges'],var_name='Finding', value_name='Excerpts')
        df = df[pd.notna(df['Excerpts'])]
        
        return df

    
    def split_excerpts(df):
        logging.info('Splitting unique learnings in each planned intervention')
        df['Excerpts_ind'] = [LineTokenizer(blanklines='discard').tokenize(x) for x in df['Excerpts']]
        df = df.explode(column = 'Excerpts_ind', ignore_index = True)
        df.drop(columns = 'Excerpts', inplace =True)
        
        ## remove strings that have less than 5 characters
        df['Excerpts_ind'] = [np.nan if pd.notna(x) and len(x) < 5 else x for x in df['Excerpts_ind']]
        df = df[pd.notna(df['Excerpts_ind'])]
        
        #catching go format (bullet point)
        df['Excerpts_ind'] = [x[2:] if x.startswith('•\t') else x for x in df['Excerpts_ind']]

        #catching other formats
        df['Excerpts_ind'] = [x[1:] if x.startswith(tuple(['-','•','▪',' '])) else x for x in df['Excerpts_ind']]

        df['Excerpts'] = [x.strip() for x in df['Excerpts_ind']]
        
        df.drop(columns = 'Excerpts_ind', inplace =True)
        
        return df
    
    final_report_interventions = split_planned_interventions(filtered_final_report)
    
    if final_report_interventions.empty:
        logging.warning('There were not found any learnings on the DREF Final Reports planned interventions')
        return None
    else:
        final_report_learnings = split_excerpts(final_report_interventions)
        return final_report_learnings


def tag_data(df, tagging, tagging_api_endpoint):
    logging.info('Tagging learnings with PER framework')
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    }
    
    url = tagging_api_endpoint
    
    df[tagging] = None
    df.reset_index(inplace= True, drop = True)
    
    for i in range (0,len(df)):
        data = "\""+df['Excerpts'].iloc[i]+"\""
        data = data.encode('utf-8')
        response = requests.post(url, headers=headers, data = data)
        if (response.status_code==201) and len(response.json()[0]['tags']) > 0:
            df.loc[i, tagging] = response.json()[0]['tags'][0]
    
    df['Institution'] = np.empty(len(df))
    
    for i in range(0,len(df)):
        if (df['PER - Component'].iloc[i] == 'Activation of Regional and International Support'):
            df.loc[i,'Institution'] = 'Secretariat'
        else:
            df.loc[i,'Institution']= 'National Society'
        
    tagged_data= df
    return tagged_data


def fetch_complementary_data(per_formcomponent, primary_sector):
    logging.info('Fetching complementary data on PER components ids, sectors ids, finding ids, organisations ids')
    def fetchUrl(field):
        return requests.get(field).json()

    def fetchField(field):
        dict_field = []
        temp_dict = requests.get(GO_API_URL+field+LIMIT_200).json()
        while temp_dict['next']:
            dict_field.extend(temp_dict['results'])
            temp_dict = fetchUrl(temp_dict['next'])
        dict_field.extend(temp_dict['results'])
        return pd.DataFrame.from_dict(dict_field)

    per_formcomponent = fetchField(per_formcomponent)
    
    go_sectors =  fetchUrl(GO_API_URL+primary_sector)
    
    dict_per = dict(zip(per_formcomponent['title'],per_formcomponent['id']))
    
    dict_sector = {item['label']: item['key'] for item in go_sectors}
    
    dict_finding = {
        'Lessons Learnt': 1,  
        'Challenges': 2
    }
    
    dict_org = {
        'Secretariat': 1,  
        'National Society': 2
    }  
    
    mapping_per = {
    "Activation of Regional and International Support": "Activation of regional and international support",
    "Affected Population Selection": "Affected population selection",
    "Business Continuity": "Business continuity",
    "Cash and Voucher Assistance": "Cash Based Intervention (CBI)",
    "Communications in Emergencies": "Communication in emergencies",
    "Coordination with Authorities": "Coordination with authorities",
    "Coordination with External Agencies and NGOs": "Coordination with External Agencies and NGOs",
    "Coordination with Local Community Level Responders": "Coordination with local community level responders",
    "Coordination with Movement": "Coordination with Movement",
    "DRM Laws, Advocacy and Dissemination": "DRM Laws, Advocacy and Dissemination",
    "Early Action Mechanisms": "Early Action Mechanisms",
    "Emergency Needs Assessment and Planning": "Emergency Needs Assessment",
    "Emergency Operations Centre (EOC)": "Emergency Operations Centre (EOC)",
    "Emergency Response Procedures (SOP)": "Emergency Response Procedures (SOPs)",
    "Finance and Admin. Policy and Emergency Procedures": "Finance and Admin policy and emergency procedures",
    "Hazard, Context and Risk Analysis, Monitoring and Early Warning": "Hazard, Context and Risk Analysis, Monitoring and Early Warning",
    "Information and Communication Technology (ICT)": "Information and Communication Technology (ICT)",
    "Information Management": "Information Management (IM)",
    "Logistics - Logistics Management": "LOGISTICS MANAGEMENT",
    "Logistics - Procurement": "PROCUREMENT",
    "Logistics - Warehouse and Stock Management": "WAREHOUSE AND STOCK MANAGEMENT",
    "Mapping of NS Capacities": "Mapping of NS capacities",
    "NS Specific Areas of Intervention": "NS-specific areas of intervention",
    "Operations Monitoring, Evaluation, Reporting and Learning": "Operations Monitoring, Evaluation, Reporting and Learning",
    "Pre-Disaster Meetings and Agreements": "Pre-disaster meetings and agreements",
    "Preparedness Plans and Budgets": "Preparedness plans and budgets",
    "Quality and Accountability": "Quality and accountability",
    "RC Auxiliary Role, Mandate and Law": "RC auxiliary role, Mandate and Law",
    "Resources Mobilisation": "Resource Mobilisation",
    "Response and Recovery Planning": "Response and recovery planning",
    "Risk Management": "Risk management",
    "Safety and Security Management": "Safety and security management",
    "Staff and Volunteer Management": "Staff and volunteer management",
    "Testing and Learning": "Testing and Learning",
    "Cooperation with Private Sector": "Cooperation with private sector",
    "Disaster Risk Management Strategy": "DRM Strategy",
    "Logistics - Supply Chain Management": "SUPPLY CHAIN MANAGEMENT",
    "Logistics - Transportation Management": "FLEET AND TRANSPORTATION MANAGEMENT",
    "Scenario Planning": "Scenario planning",
    "Civil Military Relations": "Civil Military Relations",
    "Disaster Risk Management Policy": "DRM Policy",
    "information and Communication Technology (ICT)": "Information and Communication Technology (ICT)",
    "Coordination with local community level responders": "Coordination with local community level responders",
    "Emergency Response Procedures (SOPs)": "Emergency Response Procedures (SOPs)",
    "Logistics - Transport": "FLEET AND TRANSPORTATION MANAGEMENT",
    "Unknown": None,
    "Business continuity": "Business continuity",
    "emergency Response Procedures (SOP)": "Emergency Response Procedures (SOPs)",
    "National Society Specific Areas of intervention": "NS-specific areas of intervention"
    }
    
    mapping_sector = {
    "Strategies for implementation": None,  # No direct match found
    "Disaster Risk Reduction and Climate Action": "DRR",
    "Health": "Health (public)",
    "Livelihoods and Basic Needs": "Livelihoods and basic needs",
    "Migration and Displacement": "Migration",
    "Protection, Gender and Inclusion": "PGI",
    "Shelter and Settlements": "Shelter",
    "Water Sanitation and Hygiene": "WASH",
    "Secretariat Services": None,  # No direct match found # need to bring it out as IFRC learning
    "National Society Strengthening": "NS Strengthening",
    "Water, Sanitation And Hygiene": "WASH",
    "Protection, Gender And Inclusion": "PGI",
    "Shelter Housing And Settlements": "Shelter",  
    "Livelihoods And Basic Needs": "Livelihoods and basic needs",
    "Community Engagement And Accountability": "CEA",
    "Multi-purpose Cash": "Livelihoods and basic needs",  # No direct match found
    "Risk Reduction, Climate Adaptation And Recovery": "DRR", 
    "Migration": "Migration",
    "Education": "Education",
    "Shelter and Basic Household Items":"Shelter",
    "Multi Purpose Cash": "Livelihoods and basic needs", 
    "Environmental Sustainability":None,
    "Migration And Displacement":"Migration",
    "Coordination And Partnerships":"NS Strengthening"}
    
    
    return mapping_per, dict_per, mapping_sector, dict_sector, dict_org, dict_finding
    
def format_data(df, mapping_per, dict_per, mapping_sector, dict_sector,dict_org, dict_finding):
    logging.info('Formatting data to upload to GO Operational Learning Table')
    df.loc[:,'mapped_per'] = [mapping_per[x] if pd.notna(x) else None for x in df['PER - Component']]
    df.loc[:,'id_per'] = [dict_per[x] if pd.notna(x) else None for x in df['mapped_per']]
    df.loc[:,'mapped_sector'] = [mapping_sector[x] if pd.notna(x) else None for x in df['Sector']]
    df.loc[:,'id_sector'] = [dict_sector[x] if pd.notna(x) else None for x in df['mapped_sector']]
    df.loc[:,'id_institution'] =  [dict_org[x] for x in df['Institution']]
    df.loc[:,'id_finding'] = [dict_finding[x] for x in df['Finding']]
    
    formatted_data = df[['appeal_code', 'Excerpts', 'id_per','id_sector','id_institution','id_finding']]
    
    return formatted_data


def manage_duplicates(df):
    logging.info('Managing duplicates')
    df = df.groupby(['appeal_code','Excerpts','id_finding'], as_index = False).agg(list).reset_index()
    df.drop(columns = ['index'], inplace = True)
    
    df['id_per'] = [list(set([y for y in x if pd.notna(y)])) for x in df['id_per']]
    df['id_sector'] = [list(set([y for y in x if pd.notna(y)])) for x in df['id_sector']]
    df['id_institution'] = [list(set([y for y in x if pd.notna(y)])) for x in df['id_institution']]
    
    deduplicated_data = df
    
    return deduplicated_data

def post_to_api(df, api_post_endpoint, go_auth_token_path):
    logging.info('Posting data to GO Operational Learning API')
    with open(go_auth_token_path) as json_file:
        go_authorization_token = json.load(json_file)
    
    url = api_post_endpoint
    
    myobj = {}
    for i in range(0,len(df)):
        myobj[i] = {"learning": df['Excerpts'].iloc[i],
                    "learning_validated": df['Excerpts'].iloc[i],
                    "appeal_code":df['appeal_code'].iloc[i],
                    "type":int(df['id_finding'].iloc[i]),
                    "type_validated":int(df['id_finding'].iloc[i]),
                    "sector": df['id_sector'].iloc[i],
                    "sector_validated": df['id_sector'].iloc[i],
                    "per_component": df['id_per'].iloc[i],
                    "per_component_validated": df['id_per'].iloc[i],
                    "organization": df['id_institution'].iloc[i],
                    "organization_validated": df['id_institution'].iloc[i],
                    "is_validated": False
                   }
        
    # Define a retry decorator
    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
    def post_request(x):
        response = requests.post(url, json=myobj[x],headers = go_authorization_token)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response


    for x in range(0, len(myobj)):
        try:
            response = post_request(x)
            logging.info("Response status: %s", response.status_code)
            time.sleep(1)
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
            time.sleep(5)  # Wait before retrying
        except requests.exceptions.RequestException as err:
            print(f"Request Exception: {err}")
            time.sleep(5)  # Wait before retrying

def main(go_auth_token_path):
    logging.info("Starting extracting tags for ops learnings")

    # Step 1: Fetch Data
    final_report, appeal, ops_learning = fetch_data('dref-final-report', 'appeal', 'ops-learning', go_auth_token_path)
    filtered_data = filter_final_report(final_report, appeal, ops_learning, final_report_is_published = True, appeal_is_published = True, in_ops_learning = False)
    
    if filtered_data is not None:
        # Step 2: Data Preprocessing
        split_learnings = split_rows(filtered_data)

        if split_learnings is not None:
            # Step 3: Tagging
            tagged_data = tag_data(split_learnings,'PER - Component' , CLASSIFY_URL)

            # Step 4: Post Processing
            mapping_per, dict_per, mapping_sector, dict_sector, dict_org, dict_finding = fetch_complementary_data('per-formcomponent', 'primarysector')
            formatted_data = format_data(tagged_data,mapping_per, dict_per, mapping_sector, dict_sector,dict_org, dict_finding)
            deduplicated_data = manage_duplicates(formatted_data)
    
            # Step 5: Post to API Endpoint
            post_to_api(deduplicated_data, OPS_LEARNING_URL, go_auth_token_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_tag_go_learnings.py go_authorization_token_path")
    else:
        go_auth_token_path = sys.argv[1]
        main(go_auth_token_path)