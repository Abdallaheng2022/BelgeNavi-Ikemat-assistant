from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
import requests
import json
import os 
from VB_handler import VectorDBConfig,VectorDBWrapper
"""# A DAG represents a workflow, a collection of tasks
with DAG(dag_id="demo", start_date=datetime(2022, 1, 1), schedule=None) as dag:
    # Tasks are represented as operators
    hello = BashOperator(task_id="hello", bash_command="echo hello")

    @task()
    def airflow(number_of_cases):
        params = {
            "query.cond": "Type 2 Diabetes",
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
            "fields": (
                "protocolSection.identificationModule.nctId,"
                "protocolSection.identificationModule.officialTitle,"
                "protocolSection.eligibilityModule.eligibilityCriteria,"
                "protocolSection.eligibilityModule.sex,"
                "protocolSection.eligibilityModule.minimumAge,"
                "protocolSection.eligibilityModule.maximumAge,"
                "protocolSection.contactsLocationsModule.locations,"
                "protocolSection.statusModule.overallStatus"
            ),
            "sort": "LastUpdatePostDate:desc",
            "pageSize": number_of_cases,
            "countTotal": "true",
            "format": "json",
        }
        r = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params, timeout=60)
        
        return r.json()

    # Set dependencies between tasks
    hello >> airflow(20)"""

qdrant_config = VectorDBConfig.qdrant(
       url=os.getenv("QD_END_POINT"),
       api_key=os.getenv("QD_API_KEY")
      )  



db=VectorDBWrapper(qdrant_config)
if db.connect():
        print("Qdrant connected sucess")
db.insert()