
"""
#import requests
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
env_path = os.path.join('.env')
load_dotenv(env_path)
"""
params = {
  "query.cond": "Type 2 Diabetes",
  "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
  "fields": "protocolSection.identificationModule.nctId,protocolSection.identificationModule.officialTitle,protocolSection.eligibilityModule.eligibilityCriteria,protocolSection.eligibilityModule.sex,protocolSection.eligibilityModule.minimumAge,protocolSection.eligibilityModule.maximumAge,protocolSection.contactsLocationsModule.locations,protocolSection.statusModule.overallStatus",
  "sort": "LastUpdatePostDate:desc",
  "pageSize": 20,
  "countTotal": "true",
  "format": "json"
  
}
r = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params)
#data = r.json()
with open("trail_matcher.json","w")as jsonfile:
     json.dump(r.json(),jsonfile)
"""
with open("/home/abdo/Downloads/LLMs_apps_github/trialmatch_helper_skeleton/trialmatch_helper/backend/trail_matcher.json","r") as jsonfile:
     trial_matcher=json.load(jsonfile)


trial_matcher_docs = [] 
for studies in trial_matcher['studies']:
          #print(studies['protocolSection']['contactsLocationsModule']['locations'][0].keys())
          page_content=studies['protocolSection']['eligibilityModule']['eligibilityCriteria']
          sex = studies['protocolSection']['eligibilityModule']['sex']
          try: 
             minimum_age = studies['protocolSection']['eligibilityModule']['minimumAge']    
          except: 
             minimum_age= None   
          try:   
              maximum_age=studies['protocolSection']['eligibilityModule']['maximumAge']  
          except:
              maximum_age= None    
          try: 
              locations=studies['protocolSection']['contactsLocationsModule']['locations']
          except:
              locations = None
          id=studies['protocolSection']['identificationModule']['nctId']
          title = studies['protocolSection']['identificationModule']['officialTitle']
          status = studies['protocolSection']['statusModule']
          trial_matcher_docs+=[Document(page_content=page_content,
                   metadata=dict(id=id,title=title,status=status,sex=sex,minimum_age=minimum_age,maximum_age=maximum_age,locations=locations)
                   )]
          
# Requires: export OPENAI_API_KEY=...
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vdb = FAISS.from_documents(trial_matcher_docs, embeddings)          

##['identificationModule', 'statusModule', 'eligibilityModule', 'contactsLocationsModule']
##dict_keys(['nctId', 'officialTitle'])
# dict_keys(['eligibilityCriteria', 'sex', 'minimumAge'])
#dict_keys(['eligibilityCriteria', 'sex', 'minimumAge', 'maiximumAge'])
#dict_keys(['locations'])
#dict_keys(['facility', 'status', 'city', 'state', 'zip', 'country', 'contacts', 'geoPoint'])
"""
       
  