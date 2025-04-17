# processing
# convert to the result can be used by llm_pq
def convert_to_llm_pq_result2partitions(res):
    pipeline_partition_result, bit_assignment_result = (
        res["plan"]["partition_result"],
        res["plan"]["bit_assignment"],
    )
    D = res["D"]
    # result is something like
    """
        sharding_strategy = {
        0: {},
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
        },
        2: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [8, 16]},
        },
    }
    """
    sharding_strategy = {}
    for device_rank, (layer_start, layer_end) in pipeline_partition_result.items():
        sharding_strategy[device_rank] = {}
        for layer in range(layer_start, layer_end):
            atten_idx = layer * 2
            ffn_idx = layer * 2 + 1
            atten_bit = bit_assignment_result[atten_idx]
            ffn_bit = bit_assignment_result[ffn_idx]
            # check if the bit can be replace with tc:8
            # TODO: abandon for the moment.
            # D_name = D[device_rank]
            # if atten_bit == 8:
            #     if has_tc(D_name):
            #         atten_bit = '8:tc'
            # if ffn_bit == 8:
            #     if has_tc(D_name):
            #         ffn_bit = '8:tc'
            sharding_strategy[device_rank][layer] = {
                "shard": [0, 1],
                "bits": [atten_bit, ffn_bit],
            }

    res["use_plan"] = sharding_strategy
