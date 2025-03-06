from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


def quantize_to_bit_gptq(model_id: str, quant_path: str, bits=4):  # noqa
    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",  # noqa
    ).select(range(1024))["text"]

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant_config = QuantizeConfig(bits=bits, group_size=128)
    model = GPTQModel.load(model_id, quant_config)
    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=2)
    model.save(quant_path)
