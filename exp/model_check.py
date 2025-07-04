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

from transformers import AutoConfig

config = AutoConfig.from_pretrained(args.model)
print(config.num_hidden_layers)
