from optimum.neuron import NeuronModelForCausalLM

compiler_args = {"num_cores": 8, "auto_cast_type": "fp16"}
input_shapes = {"batch_size": 4, "sequence_length": 4096}

model = NeuronModelForCausalLM.from_pretrained(
    "arcee-ai/Llama-Spark", export=True, **compiler_args, **input_shapes
)

model.save_pretrained("llama-spark-neuron")
