from llmpq.config import PQConfig
from llmpq.core import create_ada_model

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    tgt_model_path = "./tmp/Qwen2.5-14B-Instruct-ada"
    pq_config = PQConfig(
        model_id_or_path=model_id,
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        random_bits=True,
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )
    create_ada_model(pq_config, tgt_model_path)
    pq_config.save("./tmp") # export the runner scripts

            