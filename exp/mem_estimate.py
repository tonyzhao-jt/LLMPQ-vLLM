from llmpq.costmodel.core import init_cost_model
from llmpq.optimizer.algo.algo_utils import estimate_min_max_mem
from llmpq.costmodel.mem import estimate_single_layer_mem, get_device_topo_available_mem_with_order
from llmpq.costmodel.topo import create_device_topo
from transformers import AutoConfig
from llmpq.config import PQConfig
model_id = "Qwen/Qwen2.5-7B-Instruct"
global_bz = micro_bz = 256
config = AutoConfig.from_pretrained(model_id)
s = config.max_position_embeddings
s = 2048 # chunked prefill
s = s
n = 100 # not so useful
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = [2, 1]
device_topo = create_device_topo(device_names, device_numbers, use_tp=True)
comm_cost_model_dir = None 
cost_model_store_path = '/tmp/llmpq_costmodel'

model_mem_estimator, comm_cost_model, lat_cost_model, T = init_cost_model(
    config,
    global_bz,
    micro_bz,
    s,
    n,
    device_names,
    device_numbers,
    comm_cost_model_dir,
    cost_model_store_path,
)
max_mem, min_mem = estimate_min_max_mem(model_mem_estimator, layers=T, max_bit=16, min_bit=4)
embed_mem = model_mem_estimator.calculate_emb_mem()
lm_head_mem = model_mem_estimator.calculate_lm_head_mem()
mem_mesh = get_device_topo_available_mem_with_order(
    device_topo,
    model_mem_estimator,
    global_bz,
    micro_bz,
)

# heuristic
# 尽量把比较快的模型
# layers
mem_4bit = estimate_single_layer_mem(model_mem_estimator, [0,1], [4, 4])
mem_8bit = estimate_single_layer_mem(model_mem_estimator, [0,1], [8, 8])
mem_16bit = estimate_single_layer_mem(model_mem_estimator, [0,1], [16, 16])
# the mem mesh need to be further partitoned.
import pdb; pdb.set_trace()