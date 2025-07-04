from typing import Dict, List


# assign bits to the layer
def assign_uniform_bit(
    layer_shards: Dict[int, List[int]], bit: int, bit_map: Dict[int, int]
) -> Dict[int, List[int]]:
    """
    create a uniform bitwidth map for the layers
    - each layer have shards
    """
    for layer_idx, layer_shards in layer_shards.items():
        shard_num = len(layer_shards)
        bit_map[layer_idx] = [bit] * shard_num
    return bit_map
