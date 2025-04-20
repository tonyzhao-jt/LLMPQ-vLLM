from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def shard_model(model_id: str, save_path: str, layer_num: int = -1):
    config = AutoConfig.from_pretrained(model_id)
    if layer_num == -1:
        layer_num = config.num_hidden_layers
    config.num_hidden_layers = layer_num  # reduce number of layers
    reduced_model = AutoModelForCausalLM.from_config(config)
    reduced_model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_path)
