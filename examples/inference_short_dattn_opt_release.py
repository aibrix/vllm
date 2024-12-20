from transformers import pipeline, set_seed
from vllm import LLM, SamplingParams

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
#os.environ['HF_HOME'] = '/data00/jack/tmp_dattnint'
#os.environ['TRANSFORMERS_CACHE'] = '/data00/jack/tmp_dattnint'
#os.environ['HF_DATASETS_CACHE'] = '/data00/jack/tmp_dattnint'
#os.environ['TMPDIR'] = '/data00/jack/tmp_dattnint/tmp'

##os.environ['CUDACXX'] = '/usr/local/cuda-12.1/bin/nvcc' # T4
#os.environ['CUDACXX'] = '/usr/local/cuda-12.4/bin/nvcc' # L4
#os.environ['HF_TOKEN'] = 'XXXXXXXXXX'
#os.environ['HF_ENDPOINT'] = 'XXX'

##
#export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
#export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
##export no_proxy=http://code.byted.org/

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

set_seed(32)

# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192, ignore_eos=True)
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048) # short 用2k 這事情還有待商榷 這可能變成真正的middle了
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096) # llama7b default = 4096
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048) #  opt-6.7b's default max_tokens is 2048
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=600)
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200) # 512-295=217 # 因max_tokens只能大概控制decode output數量, e.g.給512結果吐出510>個.
#sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=2048) #  opt-6.7b's default max_tokens is 2048
#sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=400) #  opt-6.7b's default max_tokens is 2048

# Create an LLM.
#llm = LLM(model="THUDM/chatglm2-6b", enforce_eager=True, trust_remote_code=True) # # T4跑不動
#llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True, trust_remote_code=True, use_dattn=False) # L4 ok # T4跑不動 # good
#llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True, trust_remote_code=True, use_dattn=True) # L4 ok # T4跑不動 # good
#llm = LLM(model="meta-llama/Llama-2-7b", enforce_eager=True, trust_remote_code=True) # 沒config.json
llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, trust_remote_code=True, use_dattn=True) # patent # O
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, trust_remote_code=True) # patent # O
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, trust_remote_code=True, use_dattn=False) # patent

#llm = LLM(model="facebook/opt-2.7b", enforce_eager=True, trust_remote_code=True) # TP
#llm = LLM(model="facebook/opt-1.3b", enforce_eager=True) # TP
#llm = LLM(model="facebook/opt-125m", enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}\n\n")

