from optimum.neuron import NeuronModelForCausalLM
from transformers import AutoTokenizer

model = NeuronModelForCausalLM.from_pretrained("./llama-spark-neuron")
tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Llama-Spark")

inputs = tokenizer("What is deep-learning ?", return_tensors="pt")
outputs = model.generate(
    **inputs, max_new_tokens=128, do_sample=True, temperature=0.9, top_k=50, top_p=0.9
)
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)
