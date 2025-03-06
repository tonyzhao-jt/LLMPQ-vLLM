# from llmpq.utils import quant_to_bit_c4

# model_id = "meta-llama/Llama-3.2-1B-Instruct"

# do llm-pq on the llama model
# do profile
# pip install -v gptqmodel --no-build-isolation

# create dummy model to do profile in the vllm for the layers
# crease

# loss based method
# from datasets import load_dataset
# from gptqmodel import GPTQModel, QuantizeConfig
# quant_path = "tmp/llm_pq/Llama-3.2-1B-Instruct-gptqmodel-{}bit"
# # quantize the models
# for bits in [3, 4, 8]:
#     quant_to_bit_c4(model_id, quant_path.format(bits), bits=bits)
