from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join('.env')
load_dotenv(env_path)

def generate_embeddings(chunk_size):
            embeddings_model = OpenAIEmbeddings(chunk_size=1000)
            return embeddings_model


class FAISSProvider:
    def __init__(self,db_path,db_name,chunk_size,process_type,query):
        if process_type =="load":    
            vec_db=FAISS.load_local(folder_path=db_path, embeddings=generate_embeddings(chunk_size),index_name=db_name,allow_dangerous_deserialization=True) 
        if process_type=="retrieve":
            vec_db=FAISS.load_local(folder_path=db_path, embeddings=generate_embeddings(chunk_size),index_name=db_name,allow_dangerous_deserialization=True) 
            self.retrieve_relevant_context(query,vec_db=vec_db)  
   
    def retrieve_relevant_context(self,query, vec_db,k=5):
       if vec_db != None:
            #This function runs ANN search on the VectorDatabases
            docs = vec_db.similarity_search(query,k=k)
            return docs
       else:
            return None 


