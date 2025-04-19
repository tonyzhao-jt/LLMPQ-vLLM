from llmpq.dataset import CNNDataset, LooGLEDataset, MoonCakeDataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/Saber/llm_pq_v2/test/tmp/Llama-3.2-1B-ada")
ds_cnn = CNNDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/test-00000-of-00001.parquet',
                    tokenizer=tokenizer)
prompts = ds_cnn.sample_n_serving_prompt(10)
ds_cnn.distribution()
ds_cnn.dump_n_serving_prompts(128, './cnn.pkl')

ds_loo = LooGLEDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/longdep_qa.jsonl',
                       tokenizer=tokenizer)
prompts = ds_loo.sample_n_serving_prompt(10)
ds_loo.distribution()
ds_loo.dump_n_serving_prompts(128, './loo.pkl')

ds_mck = MoonCakeDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/mooncake_trace.jsonl',
                         tokenizer=tokenizer)
prompts = ds_mck.sample_n_serving_prompt(10)
ds_mck.distribution()
ds_mck.dump_n_serving_prompts(128, './mck.pkl')