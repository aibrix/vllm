from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, this is",
#    "The president of the United States is",
#    "The capital of France is",
#    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", use_dattn=True, enforce_eager=True)
#llm = LLM(model="facebook/opt-2.7b", use_dattn=True, enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")