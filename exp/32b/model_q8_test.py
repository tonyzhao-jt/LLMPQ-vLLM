from llmpq.config import PQConfig
from llmpq.core import create_ada_model_dummy

if __name__ == "__main__":
    MODEL="Qwen/Qwen2.5-32B-Instruct" # num hidden layers: 64
    local_path = "./tmp/Qwen2.5-32B-Instruct-q8-1-test"
    num_4bit = 0
    num_8bit = 0
    hybrid_pack =  ",".join(["8"] * 15) 
    hybrid_pack_1 =  ",".join(["8"] * 15)
    hybrid_pack_2 =  ",".join(["8"] * 15)
    hybrid_pack_3 =  ",".join(["8"] * 19) 
    num_8_tc_bit = 0
    num_16bit = 0
    bit_packs = []
    bit_packs.append(hybrid_pack)
    bit_packs.append(hybrid_pack_1)
    bit_packs.append(hybrid_pack_2)
    bit_packs.append(hybrid_pack_3)
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
        partition_config="32,32",
        adaptive_qbits=adaptive_qbits,
        num_layers=64,
        prepost_bit=8,
    )
    create_ada_model_dummy(pq_config, local_path)
    pq_config.save(local_path) # export the runner scripts

            