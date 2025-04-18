from llmpq.config import PQConfig
from llmpq.core import create_ada_model

if __name__ == "__main__":
    pq_config = PQConfig(
        model_id_or_path="meta-llama/Llama-3.2-1B",
        partition_config="2,6,6,2",
        pipeline_parallel_size=4,
        adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
        num_layers=16,
    )
    save_path = './tmp/Llama-3.2-1B-ada' # the target mixed precision model
    create_ada_model(pq_config, save_path)
    pq_config.save("./tmp") # export the runner scripts

            