from llmpq.config import PQConfig
from llmpq.core import create_ada_model_dummy

if __name__ == "__main__":
    MODEL="Qwen/Qwen2.5-7B-Instruct" # num hidden layers: 28
    local_path = "./tmp/Qwen2.5-7B-Instruct-ada-dummy"
    num_8bit = 28
    num_16bit = 0
    bit_pack_8 = ",".join(["8"] * num_8bit)
    bit_pack_16 = ",".join(["16"] * num_16bit)
    adaptive_qbits = bit_pack_8 + ',' + bit_pack_16 if num_8bit != 28 else bit_pack_8
    pq_config = PQConfig(
        model_id_or_path=MODEL,
        pipeline_parallel_size=2,
        partition_config="14,14",
        adaptive_qbits=adaptive_qbits,
        num_layers=28,
        prepost_bit=8,
    )
    
    create_ada_model_dummy(pq_config, local_path)
    pq_config.save(local_path) # export the runner scripts

            