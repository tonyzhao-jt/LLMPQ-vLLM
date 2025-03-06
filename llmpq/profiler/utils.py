from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def shard_model(model_id: str, save_path: str, layer_num: int = 1):
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = layer_num  # 减少到指定层数
    reduced_model = AutoModelForCausalLM.from_config(config)
    reduced_model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_path)
