import argparse
import pickle
from vllm import LLM, SamplingParams
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
parser.add_argument(
    "--max-concurrency",
    type=int,
    default=1,
    help="Maximum number of concurrent requests. This can be used "
    "to help simulate an environment where a higher level component "
    "is enforcing a maximum number of concurrent requests. While the "
    "--request-rate argument controls the rate at which requests are "
    "initiated, this argument will control how many are actually allowed "
    "to execute at a time. This means that when used in combination, the "
    "actual request rate may be lower than specified with --request-rate, "
    "if the server is not processing requests fast enough to keep up.")

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Name of the model.",
)

parser.add_argument(
    '--use-llmpq',
    action='store_true',
)

# --tensor-parallel-size
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=None,
    help="Number of GPUs to use for tensor parallelism. If not specified, "
    "the number of GPUs will be automatically determined based on the "
    "available GPUs.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.dataset_path
    model = args.model
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    prompts = [dp[0] for dp in data]
    max_tokens = max(dp[1] for dp in data)
    # SPDX-License-Identifier: Apache-2.0
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=max_tokens)
    tensor_parallel_size = args.tensor_parallel_size
    if args.use_llmpq:
        llm = LLM(
            model=model,
            quantization="llmpq",
            max_num_batched_tokens=2048,
            tensor_parallel_size=tensor_parallel_size,
        )  # try cpu offload
    else:
        llm = LLM(
            model=model,
            load_format='dummy',
            max_num_batched_tokens=2048,
            tensor_parallel_size=tensor_parallel_size,
        )  
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     tkn_num = len(output.outputs[0].token_ids)
        # print(
        #     f"Prompt: {prompt!r}, Generated text: {generated_text!r}, token_num: {tkn_num}"
        # )
