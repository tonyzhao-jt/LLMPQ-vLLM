from llmpq.dataset import CNNDataset, LooGLEDataset, MoonCakeDataset
ds_cnn = CNNDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/test-00000-of-00001.parquet')
prompts = ds_cnn.sample_n_serving_prompt(10)
ds_cnn.distribution()

ds_loo = LooGLEDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/longdep_qa.jsonl')
prompts = ds_loo.sample_n_serving_prompt(10)
ds_loo.distribution()

ds_mck = MoonCakeDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/mooncake_trace.jsonl')
prompts = ds_mck.sample_n_serving_prompt(10)
ds_mck.distribution()