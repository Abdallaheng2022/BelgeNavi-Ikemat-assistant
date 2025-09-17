from VB_handler import VectorDBConfig,VectorDBWrapper
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from PyPDF2 import PdfReader
import glob
from transformers import M2M100ForConditionalGeneration,M2M100Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
from docx import Document
env_path = os.path.join('.env')
load_dotenv(env_path)

MODEL_DIR = os.path.expanduser(
    "~/Downloads/LLMs_apps_github/belgeNavi/small100"  # وحّد المسار هنا
)


def extract_text_from_pdf(pdf_file):
     """
      It takes the pdf and extracts the textual content from it.
     """
     reader = PdfReader(pdf_file)
     raw_text= ""
     for idx, page in enumerate(reader.pages):
          text = page.extract_text()
          if text:
               raw_text+=text
     return raw_text 

def generate_embeddings():
    embeddings_model = OpenAIEmbeddings(chunk_size=1000)
    return embeddings_model

def chunk_the_text(raw_text):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    # Split the text
    texts = text_splitter.split_text(raw_text)
    return texts


def create_vector_faiss_database(raw_text):

    # Save the vector of texts of pdf into FAISS
    vec_db = FAISS.from_texts(raw_text, generate_embeddings())
    return vec_db

def SLMtranslator(text,trgtlang):
      """
       translate from en to tr, tr to ar, ar to tr
      """        
      model = M2M100ForConditionalGeneration.from_pretrained(MODEL_DIR)
      tokenizer = M2M100Tokenizer.from_pretrained(MODEL_DIR)
      tokenizer.src_lang = "en"
      tokenizer.tgt_lang = trgtlang
      encoded = tokenizer(text, return_tensors="pt")
      generated_tokens = model.generate(**encoded)
      translated_text=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      return translated_text    



MODEL_DIR = Path("~/Downloads/LLMs_apps_github/belgeNavi/m2m100_418M").expanduser()

tok = M2M100Tokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = M2M100ForConditionalGeneration.from_pretrained(str(MODEL_DIR), local_files_only=True)

def translate_m2m(text, src="en", tgt="tr"):
    tok.src_lang = src
    enc = tok(text, return_tensors="pt")
    gen = model.generate(**enc, max_new_tokens=256, forced_bos_token_id=tok.get_lang_id(tgt))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]
pdf_file_paths = glob.glob("backend/data/*.pdf")
docx_file_paths = glob.glob("backend/data/*.docx")
raw_text = ' '
for pdf_path in  pdf_file_paths:
   raw_text =extract_text_from_pdf(pdf_path)  
   raw_text+='\n'

for docx_path in docx_file_paths:
       doc = Document(docx_path)
       for paragraph in doc.paragraphs:
             raw_text += paragraph.text
       raw_text+="\n"                

chunck_ls=chunk_the_text(raw_text)  
chunk_ar_ls =[]
chunk_en_ls = []
chunk_tr_ls = [] 
for chunk in chunck_ls:
    chunk_ar_ls+=[translate_m2m(chunk,tgt="ar")]     
    chunk_en_ls+=[translate_m2m(chunk,tgt="en")]  
    chunk_tr_ls+=[translate_m2m(chunk,tgt="tr")] 
    
vdb_ar=create_vector_faiss_database(chunk_ar_ls)
vdb_en=create_vector_faiss_database(chunk_en_ls)
vdb_tr=create_vector_faiss_database(chunk_tr_ls)

vdb_ar.save_local(folder_path=".",index_name="belge_vdb_ar")
vdb_en.save_local(folder_path=".",index_name="belge_vdb_en")   
vdb_tr.save_local(folder_path=".",index_name="belge_vdb_tr")  

qdrant_config = VectorDBConfig.qdrant(
       url=os.getenv("QD_END_POINT"),
       api_key=os.getenv("QD_API_KEY")
      )  
db=VectorDBWrapper(qdrant_config)
if db.connect():
        print("Qdrant connected sucess")
