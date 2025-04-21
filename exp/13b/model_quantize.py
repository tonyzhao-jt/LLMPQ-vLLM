from llmpq.config import PQConfig
from llmpq.core import create_ada_model_dummy

if __name__ == "__main__":
    MODEL="NousResearch/Llama-2-13b-chat-hf" # num hidden layers: 40
    local_path = "./tmp/llama13b-int8"
    num_4bit = 0
    num_8bit = 40
    num_16bit = 0
    bit_pack_4 = ",".join(["4"] * num_4bit)
    bit_pack_8 = ",".join(["8"] * num_8bit)
    bit_pack_16 = ",".join(["16"] * num_16bit)
    adaptive_qbits = bit_pack_4 + ',' + bit_pack_8 + ',' + bit_pack_16
    adaptive_qbits = bit_pack_8
    pq_config = PQConfig(
        model_id_or_path=MODEL,
        pipeline_parallel_size=2,
        partition_config="20,20",
        adaptive_qbits=adaptive_qbits,
        num_layers=40,
        prepost_bit=8,
    )
    
    create_ada_model_dummy(pq_config, local_path)
    pq_config.save(local_path) # export the runner scripts

            