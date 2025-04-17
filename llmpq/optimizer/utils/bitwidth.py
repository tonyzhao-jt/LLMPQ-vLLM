from typing import Dict


# assign bits to the layer
def assign_uniform_bit(layers: int, bit: int, bit_map: Dict[int, int]):
    """
    create a uniform bitwidth map for the layers (int -> int)
    """
    for layer in range(layers):
        bit_map[layer] = bit
    return bit_map
