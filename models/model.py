# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import pipeline
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    pipeline, StoppingCriteria, StoppingCriteriaList
)
import torch, re


"""class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs):
        self.tok = tokenizer
        self.stop_strs = stop_strs
        self.buf = ""

    def __call__(self, input_ids, scores, **kwargs):
        # نجمع آخر نص مولّد (سريع وبسيط)
        text = self.tok.decode(input_ids[0], skip_special_tokens=False)
        # إذا ظهر أي من سلاسل الإيقاف، نوقف
        return any(s in text for s in self.stop_strs)

class AIModel:
    
    def __init__(self,model_path,tokenizer_path,type_model):
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path 
        self.model = None
        self.tokenizer=None
        self.type_model = type_model

    def get_model(self):
       
        self.model=AutoModelForCausalLM.from_pretrained(self.model_path,load_in_4bit=True)
        self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path,torch_dtype=torch.float16, device_map="auto")
         

        return 
    

    #For memory managment
    def load_bnb4_model(self,
                    keep_on_gpu: bool = True,
                    gpu_cap_gib: int = 7,
                    offload_dir: str = "./offload"):
        bnb_cfg = BitsAndBytesConfig(        
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,   # fp16 to save VRAM
        )

        tok = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        # device_map & max_memory: use int key 0 (not "cuda:0")
        device_map = "auto" if keep_on_gpu else {"": "cpu"}
        max_memory = {0: f"{gpu_cap_gib}GiB", "cpu": "48GiB"} if keep_on_gpu else None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_cfg,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_dir,
            low_cpu_mem_usage=True,
             trust_remote_code=True,
            attn_implementation="eager",   # avoids some memory spikes
            dtype=torch.float16,         # <- use dtype, not torch_dtype            
        
        ) 
       
      
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=512,
            do_sample=True,
            top_p=1.0,
            return_full_text=False    
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm,tok

"""

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
        outputs = self.model.generate(**inputs,return_dict_in_generate=True,output_scores=True,do_sample=False,max_new_tokens=3000)
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
