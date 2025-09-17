from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any,Annotated
from langfuse import Langfuse
from langfuse import get_client
import sys,os
import json
import time
from model import AIModel
from langchain.chat_models import init_chat_model 
from langgraph.graph.message import add_messages
from prompts import (get_explainer_analysis_messages,
                     get_matcher_analysis_messages,
                     get_retriever_analysis_messages,
                     get_guradials_analysis_messages,
                     get_summary_analysis_messages)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
llm=init_chat_model("gpt-4o")
from trial_vdb.trial_matcher_vdb import FAISSProvider
class TrialMatcher(TypedDict, total=False):
    query:str
    qwen_slm_constructor:AIModel
    trace: Langfuse
    patient_ctx:Any
    trials: List[Dict[str,Any]]
    descisions: Dict[str,Any]
    explainations: str
    guardrails: str
    discalmier: Dict[str,Any]
    summary: Dict


def retriever(state: TrialMatcher) -> TrialMatcher:
    user_data = state.get("query", "")
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="retriever") as span:
        patient_ctx = FAISSProvider(
            "trial_vdb",
            "trial_vdb",
            1000,
            "retrieve",
            user_data
        )

        with open(
            "trail_matcher.json",
            "r"
        ) as json_file:
            ctgovcatalog = json.load(json_file)

        retrieved_messages = get_retriever_analysis_messages(
            query=user_data,
            patient_ctx=patient_ctx,
            lang="en",
            ctgov_catalog=""
        )

        # 🔹 سجل المدخلات
        span.update(input={"query": user_data, "messages": retrieved_messages})

        answers, outputs, model = qwen_slm_constructor.get_model_output(retrieved_messages)

        try:
            trials = eval(answers.split("</think>")[-1])
        except:
            messages = [
                {"role": "system", "content": "you are json fixer any problems missed please correct it"},
                {"role": "user", "content": "please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            trials = answers.to_json()

        # 🔹 سجل المخرجات
        span.update(output={"trials": trials, "patient_ctx": patient_ctx})

        return {"trials": trials, "patient_ctx": patient_ctx}
       

def matcher(state: TrialMatcher):
    user_data = state.get("query", "")
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    trials = state.get("trials", "")
    patient_ctx = state.get("patient_ctx", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="matcher") as span:
        matcher_messages = get_matcher_analysis_messages(
            query=user_data,
            patient_ctx=patient_ctx,
            lang="en",
            trials=trials
        )

        # 🔹 سجل المدخلات
        span.update(input={"query": user_data, "trials": trials, "patient_ctx": patient_ctx})

        answers, outputs, model = qwen_slm_constructor.get_model_output(matcher_messages)

        try:
            decisions = eval(answers.split("</think>")[-1])
        except:
            messages = [
                {"role": "system", "content": "you are json fixer any problems missed please correct it"},
                {"role": "user", "content": "please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            decisions = answers.to_json()

        # 🔹 سجل المخرجات
        span.update(output={"decisions": decisions})

        return {"decisions": decisions}


def explainer(state: TrialMatcher):
       user_data=state.get("query","")
       qwen_slm_constructor =state.get("qwen_slm_constructor","")
       trials=state.get("trials","")
       patient_ctx = state.get("patient_ctx","")
       decisions = state.get("descisions","")
       explainations_messages=get_explainer_analysis_messages( lang="en", query=user_data,patient_ctx=patient_ctx, trials=trials,decisions=decisions)
       answers,outputs,model=qwen_slm_constructor.get_model_output(explainations_messages)
       try:
            explainations=eval(answers.split("</think>")[-1])
       except: 
             messages =  [{"role":"system","content":"you are json fixer any problems missed please correct it"},{"role":"user","content":"please could you fix this json structure"}]
             answers=llm.invoke(messages)
             answers=llm.invoke(messages)
             explainations=answers.to_json()
       return {"explainations":explainations}

def validator(state: TrialMatcher):
    user_data = state.get("query", "")
    qwen_slm_constructor = state.get("qwen_slm_constructor", "")
    trials = state.get("trials", "")
    patient_ctx = state.get("patient_ctx", "")
    decisions = state.get("descisions", "")
    explainations = state.get("explainations", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="validator") as span:
        guardrails_messages = get_guradials_analysis_messages(
            query=user_data,
            lang="en",
            patient_ctx=patient_ctx,
            trials=trials,
            decisions=decisions,
            explanations=explainations
        )

        # 🔹 سجل المدخلات
        span.update(input={
            "query": user_data,
            "trials": trials,
            "patient_ctx": patient_ctx,
            "decisions": decisions,
            "explanations": explainations
        })

        start = time.time()
        answers, outputs, model = qwen_slm_constructor.get_model_output(guardrails_messages)
        end = time.time()

        qwen_slm_constructor.free_cuda(outputs, model)

        try:
            guardrails = eval(answers.split("</think>")[-1])
        except:
            messages = [
                {"role": "system", "content": "you are json fixer any problems missed please correct it"},
                {"role": "user", "content": "please could you fix this json structure"}
            ]
            answers = llm.invoke(messages)
            guardrails = answers.to_json()

        # 🔹 سجل المخرجات + زمن التنفيذ
        span.update(
            output={"guardrails": guardrails},
            metadata={"latency_seconds": round(end - start, 2)}
        )

        return {"guardrails": guardrails}

def summarize(state: TrialMatcher):
    user_data = state.get("query", "")
    trials = state.get("trials", "")
    patient_ctx = state.get("patient_ctx", "")
    decisions = state.get("descisions", "")
    explainations = state.get("explainations", "")
    guardrails = state.get("guardrails", "")
    langfuse = state.get("trace", "")

    with langfuse.start_as_current_span(name="summarize") as span:
        summary_messages = get_summary_analysis_messages(
            query=user_data,
            lang="en",
            patient_ctx=patient_ctx,
            trials=trials,
            decisions=decisions,
            explanations=explainations,
            guardrials=guardrails
        )

        # 🔹 سجل المدخلات
        span.update(input={
            "query": user_data,
            "trials": trials,
            "patient_ctx": patient_ctx,
            "decisions": decisions,
            "explanations": explainations,
            "guardrails": guardrails
        })

        start = time.time()
        summary = llm.invoke(summary_messages)
        end = time.time()

        try:
            summary_json = eval(summary.content)
        except Exception as e:
            summary_json = {"error": f"Failed to eval summary: {str(e)}", "raw": summary.content}

        # 🔹 سجل المخرجات + زمن التنفيذ
        span.update(output=summary_json, metadata={"latency_seconds": round(end - start, 2)})

        return {"summary": summary_json}

def build_graph():
       graph = StateGraph(TrialMatcher)
       graph.add_node("retriever",retriever)
       graph.add_node("matcher",matcher)
       graph.add_node("explainer", explainer)
       graph.add_node("validator",validator)
       graph.add_node("summarize",summarize)
       graph.set_entry_point("retriever")
       graph.add_edge("retriever","matcher")
       graph.add_edge("matcher","explainer")
       graph.add_edge("explainer","validator")
       graph.add_edge("validator","summarize")
       graph.add_edge("summarize",END)


       app=graph.compile()
       return app

def run():
    
    print("Trial_matcher assistant")
    print("Type 'exit' to quit\n")

    while True:
      user_input = input("Ask me anything: ")
      if user_input.lower() == "exit":
           print("Bye")
           break
      qwen_slm_constructor = AIModel(model_path="Qwen3-4B",tokenizer_path="Qwen3-4B")
      state = {
            "user_data": user_input,
            "qwen_slm_constructor":qwen_slm_constructor,
            "lang": "en",
            "patient_ctx":None,
            "trials": None,
            "descisions":None,
            "explainations": None,
            "guardrails":None,
            "summary":None
           }
      
      print("trial and matcher Processing....\n")
      final_state = app.invoke(state)
      if final_state.get("summary"):
          final_answer= final_state.get("summary")
          return final_answer      
def run_chatbot(user_query,app):
       
        qwen_slm_constructor = AIModel(model_path="/app/Qwen3-4B",tokenizer_path="/app/Qwen3-4B")
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SERCET_KEY")
            ) 
    
        state = {
                "query": user_query,
                "qwen_slm_constructor":qwen_slm_constructor,
                "trace":langfuse,
                "lang": "en",
                "patient_ctx":None,
                "trials": None,
                "descisions":None,
                "explainations": None,
                "guardrails":None,
                "summary":None
                }

        print("trial and matcher Processing....\n")
        final_state = app.invoke(state)
        if final_state.get("summary"):
            final_answer= final_state.get("summary")
            return final_answer        

