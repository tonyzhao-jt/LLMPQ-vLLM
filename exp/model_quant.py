from llmpq.config import PQConfig
from llmpq.core import create_ada_model

if __name__ == "__main__":
    model_lists = [
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Llama-3.3-70B-Instruct",
    ]
    for model_id in model_lists:
        tgt_model_path = f"/mnt/bd/juntao-model-storage/{model_id}-ada"
        pq_config = PQConfig(
            model_id_or_path=f"/mnt/bd/juntao-model-storage/{model_id}",
            work_dir="/mnt/bd/juntao-model-storage/llmpq",
            ref_16_model_path=f"/mnt/bd/juntao-model-storage/{model_id}",
            partition_config="2,6,6,2",
            pipeline_parallel_size=4,
            random_bits=True,
            adaptive_qbits="4,4" + ",8,8,8,8,8,8" + ",8,8,8,8,8,8" + ",16,8",
            num_layers=16,
        )
        create_ada_model(pq_config, tgt_model_path)
        pq_config.save(tgt_model_path)  # export the runner scripts
