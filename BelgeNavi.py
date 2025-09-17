from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any,Annotated
from langgraph.graph.message import add_messages
import sys, os
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse import get_client
import time

from prompts import (get_citer_analysis_messages,
                    get_composer_analysis_messages,
                    get_classifer_analysis_messages,
                    get_retriever_analysis_messages,
                    get_guardrails_analysis_messages
                    ,get_form_filler_analysis_messages
                    ,get_summary_analysis_messages)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.model import AIModel
from backend.belge_vector_dbs_ops.belge_local_db import FAISSProvider
from langchain.chat_models import init_chat_model

load_dotenv()
llm = init_chat_model("gpt-4o")


#meta_midllm_constructor = AIModel(model_path="/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/models/Meta-Llama-3-8B-Instruct",tokenizer_path="/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/models/Meta-Llama-3-8B-Instruct",type_model="meta")

#meta_midllm,tok=meta_midllm_constructor.load_bnb4_model(keep_on_gpu=False)



class BelgeState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    trace: Langfuse
    query: str
    qwen_slm_constructor: AIModel
    lang: str
    classified_json:Dict[str,Any]
    top_chunks: Dict[str,Any]
    extra_knwedge: Any
    citations: Dict
    checklist_composer: Dict[str,Any]
    form_filler:Dict[str,Any]
    guardrails: Dict[str,Any]
    summary: str
    
    


def classifier(state: BelgeState):
       langfuse=state.get("trace","")      
       query=state.get("query","")
       lang=state.get("lang","") 
       messages=get_classifer_analysis_messages(query,lang)
       with langfuse.start_as_current_span(name="classifier") as span:
            span.update(input=messages)
            start = time.time()
            qwen_slm_constructor = state.get("qwen_slm_constructor","")
            end = time.time() 
            answers,outputs,model=qwen_slm_constructor.get_model_output(messages)
            span.update(output=answers,metadata={"latency_seconds": round(end - start, 2)})
            
            #qwen_slm_constructor.free_cuda(outputs,model)
            try:
                  classified_json=eval(answers.split("</think>")[-1].strip())
            except: 
                  messages = [{{"role":"system","content":"you are json fixer any problems missed please correct it"},{"role":"user","content":"please could you fix this json structure"}}]
                  llm.invoke(messages)    

            return {"classified_json":classified_json}
def retriever(state:BelgeState):
    classified_json = state.get("classified_json", {})
    query = state.get("query", "")
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    langfuse = state.get("trace", "")  

    with langfuse.start_as_current_span(name="retriever") as span:
        # 🔹 سجل المدخلات
        span.update(input={"query": query, "classified_json": classified_json})

        top_chunks = None
        if classified_json['lang'] == 'tr':
            top_chunks = FAISSProvider(
                "/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/belge_vector_dbs_ops/belge_vdbs_store/",
                "belge_vdb_tr", 1000, 'retrieve', query
            )
        elif classified_json['lang'] == 'en':
            top_chunks = FAISSProvider(
                "/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/belge_vector_dbs_ops/belge_vdbs_store",
                "belge_vdb_en", 1000, 'retrieve', query
            )
        elif classified_json['lang'] == 'ar':
            top_chunks = FAISSProvider(
                "/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/belge_vector_dbs_ops/belge_vdbs_store/",
                "belge_vdb_ar", 1000, 'retrieve', query
            )            

        top_chunks_messages = get_retriever_analysis_messages(
            user_question=query,
            classified_json=classified_json,
            extra_knowledge=top_chunks
        )
        top_chunks_json, outputs, model = qwen_slm_constructor.get_model_output(top_chunks_messages)

        try:
            top_chunks_json = eval(top_chunks_json.split("</think>")[-1])
        except: 
            messages = [
                {"role":"system","content":"you are json fixer any problems missed please correct it"},
                {"role":"user","content":"please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            top_chunks_json = answers.to_json()  

        # 🔹 سجل المخرجات
        span.update(
            output=top_chunks_json  
        )

        return {"top_chunks": top_chunks_json, "extra_knowledge": top_chunks}

def citer(state: BelgeState):
    classified_json = state.get("classified_json", {})
    query = state.get("query", "")
    top_chunks = state.get("top_chunks", {})
    extra_knowledge = state.get("extra_knowledge", [])
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="citer") as span:
        citer_system_messages = get_citer_analysis_messages(
            user_question=query,
            lang=classified_json['lang'],
            top_chunks_jsonl=top_chunks,
            extra_knowledge=extra_knowledge
        )

        # 🔹 سجل المدخلات
        span.update(input=citer_system_messages)

        answers, outputs, model = qwen_slm_constructor.get_model_output(citer_system_messages) 

        try:
            citations = eval(answers.split("</think>")[-1])
        except: 
            messages = [
                {"role":"system","content":"you are json fixer any problems missed please correct it"},
                {"role":"user","content":"please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            citations = answers.to_json()    

        # 🔹 سجل المخرجات
        span.update(output=citations)

        return {"citations": citations}

def checklist_composer(state: BelgeState):
    classified_json = state.get("classified_json", {})
    citations = state.get("citations", {})
    query = state.get("query", "")
    top_chunks = state.get("extra_knowledge", {})
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="checklist_composer") as span:
        composer_system_messages = get_composer_analysis_messages(
            user_question=query,
            lang=classified_json['lang'],
            classified_json=classified_json,
            cited_bullets_jsonl=citations,
            extra_knowledge=top_chunks
        )

        # 🔹 سجل المدخلات
        span.update(input=composer_system_messages)

        answers, outputs, model = qwen_slm_constructor.get_model_output(composer_system_messages)

        try:
            checklist_composer = eval(answers.split("</think>")[-1])
        except: 
            messages = [
                {"role":"system","content":"you are json fixer any problems missed please correct it"},
                {"role":"user","content":"please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            checklist_composer = answers.to_json()    

        # 🔹 سجل المخرجات
        span.update(output=checklist_composer)

        return {"checklist_composer": checklist_composer}

def form_filler(state: BelgeState):
    classified_json = state.get("classified_json", {})
    query = state.get("query", "")
    checklist_composer = state.get("checklist_composer", "")
    extra_knowledge = state.get("extra_knowledge", "")
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="form_filler") as span:
        form_filler_analysis_messages = get_form_filler_analysis_messages(
            user_question=query,
            lang=classified_json['lang'],
            extra_knowledge=extra_knowledge,
            classified_json=classified_json,
            checklist_json=checklist_composer
        )

        # 🔹 سجل المدخلات
        span.update(input=form_filler_analysis_messages)

        answers, outputs, model = qwen_slm_constructor.get_model_output(form_filler_analysis_messages)

        try:
            form_filler = eval(answers.split("</think>")[-1])
        except: 
            messages = [
                {"role":"system","content":"you are json fixer any problems missed please correct it"},
                {"role":"user","content":"please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            form_filler = answers.to_json()   

        # 🔹 سجل المخرجات
        span.update(output=form_filler)

        return {"form_filler": form_filler}

def guardrails(state: BelgeState):
    classified_json = state.get("classified_json", {})
    citations = state.get("citations", {})
    query = state.get("query", "")
    checklist_composer = state.get("checklist_composer", "")
    extra_knowledge = state.get("extra_knowledge", "")
    form_filler = state.get("form_filler", "")
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="guardrails") as span:
        guardrails_messages = get_guardrails_analysis_messages(
            query=query,
            classified_json=classified_json,
            citer_json=citations,
            composer_json=checklist_composer,
            form_json=form_filler
        )

        # 🔹 سجل المدخلات
        span.update(input=guardrails_messages)

        answers, outputs, model = qwen_slm_constructor.get_model_output(guardrails_messages)

        try:
            guardrails_json = eval(answers.split("</think>")[-1])
        except: 
            messages = [
                {"role":"system","content":"you are json fixer any problems missed please correct it"},
                {"role":"user","content":"please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            guardrails_json = answers.to_json()  

        # 🔹 سجل المخرجات
        span.update(output=guardrails_json)

        return {"guardrails_json": guardrails_json}
def summarize(state: BelgeState):
    classified_json = state.get("classified_json", {})
    citations = state.get("citations", {})
    query = state.get("query", "")
    checklist_composer = state.get("checklist_composer", "")
    extra_knowledge = state.get("extra_knowledge", "")
    form_filler = state.get("form_filler", "")
    guardrails = state.get("guardrails", "")
    top_chunks_retrieved = state.get("top_chunks", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="summarize") as span:
        summary_messages = get_summary_analysis_messages(
            query=query,
            classified_json=classified_json,
            citations=citations,
            checklist_composer=checklist_composer,
            extra_knowledge=extra_knowledge,
            form_filler=form_filler,
            guardrails=guardrails,
            top_chunks=top_chunks_retrieved
        )

        # 🔹 سجل المدخلات
        span.update(input=summary_messages)

        summary = llm.invoke(summary_messages)

        try:
            summary_json = eval(summary.content)
        except Exception as e:
            summary_json = {"error": f"Failed to eval summary: {str(e)}", "raw": summary.content}

        # 🔹 سجل المخرجات
        span.update(output=summary_json)

        return {"summary": summary_json}

"""
def live_refresher(state: BelgeState):
       return 
def need_refresh(stat: BelgeState):
       return 
"""
def build_app():      
      graph = StateGraph(BelgeState)
      graph.add_node("classifier",classifier)
      graph.add_node("retriever",retriever)
      graph.add_node("citer", citer)
      graph.add_node("checklist_composer",checklist_composer)
      graph.add_node("form_filler",form_filler)
      graph.add_node("guardrails",guardrails)
      graph.add_node("summarize",summarize)
      graph.set_entry_point('classifier')
      graph.add_edge("classifier","retriever")
      """
      graph.add_conditional_edges(
            "retriever",
            need_refresh,
            {
            "LIVE": "live_referesher",
            "CITE": "citer"   

            }

      )
      """
      #graph.add_edge("live_referesher","retriever")
      graph.add_edge("retriever","citer")
      graph.add_edge("citer","checklist_composer",)
      graph.add_edge("checklist_composer","form_filler")
      graph.add_edge("form_filler","guardrails")
      graph.add_edge("guardrails","summarize")
      graph.add_edge("summarize",END)

      app=graph.compile()
      
      return app

def run_chatbot():
    
    print("BelgeNavi For Ikamet assistant")
    print("Type 'exit' to quit\n")

    while True:
      user_input = input("Ask me anything: ")
      if user_input.lower() == "exit":
           print("Bye")
           break
      qwen_slm_constructor = AIModel(model_path="/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/models/Qwen3-4B",tokenizer_path="/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/models/Qwen3-4B")
      langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SERCET_KEY")
            )
      
      langfuse = get_client()

      state = {
            "messages":[{"role":"user","content":user_input}],
            "trace":langfuse,
            "query": user_input,
            "qwen_slm_constructor":qwen_slm_constructor,
            "lang": None,
            "classified_json":None,
            "top_chunks": None,
            "citations": None,
            "extra_knowledge":None,
            "checklist_composer":None,
            "form_filler":None,
            "summary":None
           }
      
      print("BelgeNavi Processing....\n")
      final_state = app.invoke(state)
      if final_state.get("summary"):
          final_answer= final_state.get("summary")["bullets"]
          for idx,bullet in enumerate(final_answer):
                print(f"{idx+1}- {bullet}.")
      print("-"*80)      
def run(user_input,app):
      qwen_slm_constructor = AIModel(model_path="/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/models/Qwen3-4B",tokenizer_path="/home/abdo/Downloads/LLMs_apps_github/belgeNavi/backend/models/Qwen3-4B")
      langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SERCET_KEY")
            )

      langfuse = get_client()

      state = {
            "messages":[{"role":"user","content":user_input}],
            "trace":langfuse,
            "query": user_input,
            "qwen_slm_constructor":qwen_slm_constructor,
            "lang": None,
            "classified_json":None,
            "top_chunks": None,
            "citations": None,
            "extra_knowledge":None,
            "checklist_composer":None,
            "form_filler":None,
            "summary":None
      }

      print("BelgeNavi Processing....\n")
      final_state = app.invoke(state)
      if final_state.get("summary"):
            final_answer= final_state.get("summary")["bullets"]
      for idx,bullet in enumerate(final_answer):
            print(f"{idx+1}- {bullet}.")      
      print("-"*80)                  
      return final_state['summary']
if __name__ == "__main__":
      run_chatbot()



