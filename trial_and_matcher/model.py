# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
from transformers import pipeline
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    pipeline, StoppingCriteria, StoppingCriteriaList
)
import torch, re


class AIModel:
    
    def __init__(self,model_path,tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path 
        self.model = None
        self.tokenizer=None
        self.model=AutoModelForCausalLM.from_pretrained(self.tokenizer_path,load_in_4bit=True)
        self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)

    def get_model_output(self,messages):
        
        inputs =  self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                  ).to(self.model.device)  
        outputs = self.model.generate(**inputs,return_dict_in_generate=True,output_scores=True,do_sample=False,max_new_tokens=4000)
        sequences = outputs.sequences  # tensor of ids
        text = self.tokenizer.decode(sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text,outputs,self.model
    
    def free_cuda(*objs):
        import gc, torch
        # 1) Delete anything holding CUDA tensors (models, outputs, caches)
        for o in objs:
            try:
                if hasattr(o, "to"):  # model or module
                    o.to("cpu")
            except Exception:
                pass
            try:
                del o
            except Exception:
                pass
        # 2) GC + CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
