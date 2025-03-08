# perform the analyze on the trace 
from llmpq.profiler import parse_module_avg_cost
if __name__ == "__main__":
    trace_path = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed/Llama_3.2_1B_Instruct_sharded-gptq-4-1-pt.trace.json"
    module_cnt = {
        'VocabParallelEmbedding': 2,
        'LlamaDecoderLayer': 2,
        'LogitsProcessor': 1
    }
    cost_table, gpu_name = parse_module_avg_cost(module_cnt, trace_path)
    print(cost_table)
    print(gpu_name)
