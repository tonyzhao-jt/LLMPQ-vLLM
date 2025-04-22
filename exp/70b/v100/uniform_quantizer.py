from llmpq.config import PQConfig
from llmpq.core import create_ada_model_dummy

if __name__ == "__main__":
    # MODEL="Qwen/Qwen2-72B" # num hidden layers: 80
    # local_path = "./tmp/QWen-72B-8bit"
    # local_path = "./tmp/QWen-72B-4bit"
    MODEL="osllmai-community/Llama-3.3-70B-Instruct"
<<<<<<<< HEAD:exp/70b/A100/uniform_quantizer.py
    local_path = "./tmp/Llama-3.3-70B-4bit"
    # local_path = "./tmp/Llama-3.3-70B-8bit"
    # MODEL='meta-llama/Llama-2-70b-chat-hf'
    # local_path = "./tmp/Llama-2-70B-4bit"
    # local_path = "./tmp/Llama-2-70B-8bit"
    # num_4bit = 0
    # num_8bit = 80
    num_4bit = 80
    num_8bit = 0
    # num_8_tc_bit = 40
========
    # local_path = "./tmp/Llama-3.3-70B-4bit"
    # local_path = "./tmp/Llama-3.3-70B-8bit"
    MODEL='meta-llama/Llama-2-70b-chat-hf'
    # local_path = "./tmp/Llama-2-70B-4bit"
    local_path = "./tmp/Llama-2-70B-8bit"
    num_4bit = 0
    num_8bit = 80
>>>>>>>> a675fb2abdb82fbb55ddda20a556c843f4a347d2:exp/70b/v100/uniform_quantizer.py
    num_8_tc_bit = 0
    num_16bit = 0
    bit_packs = []
    if num_4bit > 0:
        bit_pack_4 = ",".join(["4"] * num_4bit)
        bit_packs.append(bit_pack_4)
    if num_8bit > 0:
        bit_pack_8 = ",".join(["8"] * num_8bit)
        bit_packs.append(bit_pack_8)
    if num_8_tc_bit > 0:
        bit_pack_8_tc = ",".join(["8-tc"] * num_8_tc_bit)
        bit_packs.append(bit_pack_8_tc)
    if num_16bit > 0:
        bit_pack_16 = ",".join(["16"] * num_16bit)
        bit_packs.append(bit_pack_16)
    adaptive_qbits = ','.join(bit_packs)
    pq_config = PQConfig(
        model_id_or_path=MODEL,
        pipeline_parallel_size=2,
        partition_config="40,40",
        adaptive_qbits=adaptive_qbits,
        num_layers=80,
        prepost_bit=8,
    )
    create_ada_model_dummy(pq_config, local_path)
    pq_config.save(local_path) # export the runner scripts

            