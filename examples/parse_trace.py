# read the cost, dump to csv
from llmpq.profiler import parse_module_avg_cost
from llmpq.utils.save import save_with_pickle
if __name__ == "__main__":
    module_cnt = {
        'VocabParallelEmbedding': 2,
        'LlamaDecoderLayer': 2,
        'LogitsProcessor': 1
    }
    trace_paths = [
        '/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed/Llama_3.2_1B_Instruct_sharded-bitsandbytes-8-1-pt.trace.json',
        # '/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed/Llama_3.2_1B_Instruct_sharded-gptq-4-1-pt.trace.json',
        # '/opt/tiger/Saber/llm_pq_v2/examples/tmp/vllm_profile_parsed/Llama_3.2_1B_Instruct_sharded-gptq-8-1-pt.trace.json'
    ]
    bits = [
        '8-tc',
        # '4',
        # '8'
    ]
    cost_table_dict = {}
    for trace_path, bit in zip(trace_paths, bits):
        cost_table, gpu_name = parse_module_avg_cost(module_cnt, trace_path)
        cost_table_dict[bit] = cost_table
    save_with_pickle(cost_table_dict, 'cost_table_dict_tc.pkl', './')
    print(cost_table)
    print(gpu_name)
