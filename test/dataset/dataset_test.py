from llmpq.dataset import CNNDataset, LooGLEDataset, MoonCakeDataset
from transformers import AutoTokenizer, AutoConfig
import argparse
parser = argparse.ArgumentParser()
# model
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Name of the model.",
)
args = parser.parse_args()
model_id = args.model
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
max_seq_len = config.max_position_embeddings
ds_cnn = CNNDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/test-00000-of-00001.parquet',
                    tokenizer=tokenizer)
prompts = ds_cnn.sample_n_serving_prompt(10)
ds_cnn.distribution()
ds_cnn.dump_n_serving_prompts(256, './cnn.pkl', max_seq_len=max_seq_len)

ds_loo = LooGLEDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/shortdep_qa.jsonl',
                       tokenizer=tokenizer)
prompts = ds_loo.sample_n_serving_prompt(10)
ds_loo.distribution()
ds_loo.dump_n_serving_prompts(256, './loo.pkl', max_seq_len=max_seq_len)

# ds_mck = MoonCakeDataset(data_files='/opt/tiger/Saber/llm_pq_v2/test/dataset/mooncake_trace.jsonl',
#                          tokenizer=tokenizer)
# prompts = ds_mck.sample_n_serving_prompt(10)
# ds_mck.distribution()
# ds_mck.dump_n_serving_prompts(128, './mck.pkl')