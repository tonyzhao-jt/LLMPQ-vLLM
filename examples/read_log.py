from llmpq.profiler.indicator import LossIndicator  # noqa
from llmpq.profiler.indicator import MixPrecisionIndicatorContainer  # noqa

# take a quantized model local path, read the quant_log.csv and get loss
model_path = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded-gptq-4"  # noqa
model_path_8 = "/opt/tiger/Saber/llm_pq_v2/examples/tmp/llm_pq/Llama_3.2_1B_Instruct_sharded-gptq-8"  # noqa

container = MixPrecisionIndicatorContainer()
container.add(4, LossIndicator(model_path))
container.add(8, LossIndicator(model_path_8))
container.layer_wise()
container.module_wise()
