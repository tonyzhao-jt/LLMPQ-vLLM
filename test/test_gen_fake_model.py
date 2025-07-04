from llmpq.config import PQConfig
from llmpq.core import create_ada_model

if __name__ == "__main__":
    pq_config = PQConfig(
        model_id_or_path="meta-llama/Llama-3.2-1B",
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        random_bits=True,
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )
    create_ada_model(pq_config, "./tmp/llama-3.2-1b-ada")
    pq_config.save("./tmp/llama-3.2-1b-ada")  # export the runner scripts
