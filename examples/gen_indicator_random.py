from llmpq.profiler.indicator import MixPrecisionIndicatorContainer  # noqa
from llmpq.profiler.indicator import RandomIndicator  # noqa

# take a quantized model local path, read the quant_log.csv and get loss
model_id = "meta-llama/Llama-3.2-1B-Instruct"


container = MixPrecisionIndicatorContainer()
container.add(4, RandomIndicator(model_id, bit=4))
container.add(8, RandomIndicator(model_id, bit=8))
# import pdb
# pdb.set_trace()
container.layer_wise()
# container.module_wise()

container.store("random_ind.json")
n_container = MixPrecisionIndicatorContainer.load("random_ind.json")
