from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_id = "afm_45b_ov_int8"
model = OVModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("What does Arcee AI do?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=64)
text = tokenizer.batch_decode(outputs)[0]
print(text)

