from llmpq.utils import QUANTIZATION_REGISTRY, quantize_model, save_ckpt_dummy
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
save_ckpt_dummy(model_id, "./Meta-Llama-3.1-8B-Instruct")
quantize_model("gptq", "./Meta-Llama-3.1-8B-Instruct", "./Meta-Llama-3.1-8B-Instruct-gptq8", 8)