
from datetime import datetime,timedelta
from pathlib import Path
import json, requests
from airflow import DAG
from airflow.decorators import dag,task
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings   # <- correct import
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv 
from VB_handler import VectorDBConfig,VectorDBWrapper
#Instruction: please move this code into the airflow/dag your local directory
env_path = os.path.join('/home/abdo/airflow/dags/.env')
load_dotenv(env_path)

BASE = Path(os.getenv("AIRFLOW_HOME", str(Path.home() / "airflow"))) / "dags" 
BASE.mkdir(parents=True, exist_ok=True)

with DAG(dag_id="demo", start_date=datetime(2022, 1, 1), schedule=None) as dag:
    @task()
    def fetch_data(number_of_cases: int) -> str:
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

    @task()
    def parse_data(raw_json: dict) -> str:
        studies = raw_json.get("studies", []) or []
        docs_slim = []

        for st in studies:
            ps = st.get("protocolSection", {}) or {}
            elig = ps.get("eligibilityModule", {}) or {}
            ident = ps.get("identificationModule", {}) or {}
            status = ps.get("statusModule", {}) or {}
            contacts = ps.get("contactsLocationsModule", {}) or {}

            content = elig.get("eligibilityCriteria") or ""
            if not content:
                continue

            docs_slim.append({
                "page_content": content,
                "metadata": {
                    "id": ident.get("nctId"),
                    "title": ident.get("officialTitle"),
                    "status": status.get("overallStatus"),
                    "sex": elig.get("sex"),
                    "minimum_age": elig.get("minimumAge"),
                    "maximum_age": elig.get("maximumAge"),
                    "locations": contacts.get("locations"),
                },
            })

       
        return docs_slim

    @task()
    def create_vdb(docs_input) -> dict:
        """
        docs_input: إمّا مسار ملف JSON (str) أو list[dict] بالشكل:
          [{"page_content": "...", "metadata": {...}}, ...]
        """
        # 1) حمّل الداتا
        if isinstance(docs_input, str):
            docs_path = Path(docs_input)
            raw = json.loads(docs_path.read_text(encoding="utf-8"))
        elif isinstance(docs_input, list):
            raw = docs_input
        else:
            raise TypeError(f"create_vdb expects str(path) or list[dict], got {type(docs_input)}")

        if not raw:
            raise ValueError("No documents to index (empty input).")

        # تحقّق من البنية
        if not isinstance(raw, list) or not isinstance(raw[0], dict) or "page_content" not in raw[0]:
            raise ValueError("docs must be list[dict] with 'page_content' and 'metadata' keys.")

        documents = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw]

        # 2) Embeddings
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        # 3) FAISS
        vdb = FAISS.from_documents(documents, embeddings)

        # 4) حفظ بمسار مطلق مضمُون
        index_name = "trial_vdb"
        out_dir = BASE / index_name
        vdb.save_local(folder_path=str(out_dir), index_name=index_name)

        return {"dir": str(out_dir), "index_name": index_name}

    @task
    def save_vdb(vdb_info: dict) -> str:
        # Already saved in create_vdb; here you could upload to S3/GCS, etc.
        print(f"Vector DB saved at {vdb_info['dir']} (index: {vdb_info['index_name']})")
        return vdb_info["dir"]
     
    raw = fetch_data(20)
    parsed = parse_data(raw)
    vdb_info = create_vdb(parsed)
    save_vdb(vdb_info)
  

