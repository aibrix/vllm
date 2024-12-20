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

# Sample prompts.
prompts = [
    "I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people’s attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it’s children then you can talk about animals; If it’s adults then history-based tales might engage them better etc. My first request is “I need an interesting story on perseverance.",
    "I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is “I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30.”",
    "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is “I am in Istanbul/Beyoğlu and I want to visit only museums.”",
    "Please summarize the following paragraphs in less than 20 words: Large language models trained on massive text collections have shown surprising emergent capabilities to generate text and perform zero- and few-shot learning. While in some cases the public can interact with these models through paid APIs, full model access is currently limited to only a few highly resourced labs. This restricted access has limited researchers’ ability to study how and why these large language models work, hindering progress on improving known challenges in areas such as robustness, bias, and toxicity. We present Open Pretrained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We train the OPT models to roughly match the performance and sizes of the GPT-3 class of models, while also applying the latest best practices in data collection and efficient training. Our aim in developing this suite of OPT models is to enable reproducible and responsible research at scale, and to bring more voices to the table in studying the impact of these LLMs. Definitions of risk, harm, bias, and toxicity, etc., should be articulated by the collective research community as a whole, which is only possible when models are available for study.",
]

set_seed(32)

# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192, ignore_eos=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048) #  opt-6.7b's default max_tokens is 20486
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096) # llma-7b's default max_tokens is 4096
#sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=2048) #  opt-6.7b's default max_tokens is 2048
#sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=250) #  opt-6.7b's default max_tokens is 2048

# Create an LLM.
#llm = LLM(model="THUDM/chatglm2-6b", enforce_eager=True, trust_remote_code=True) # # T4跑不動
#llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True, trust_remote_code=True, use_dattn=False) # L4 O # T4跑不動 (good)
#llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True, trust_remote_code=True, use_dattn=True) # L4 O # T4跑不動 (good)
#llm = LLM(model="meta-llama/Llama-2-7b-hf", trust_remote_code=True)
#llm = LLM(model="meta-llama/Llama-2-7b", enforce_eager=True, trust_remote_code=True) # 沒config.json
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, trust_remote_code=True, use_dattn=False)
llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, trust_remote_code=True, use_dattn=True)

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
