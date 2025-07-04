from typing import Dict, List, Tuple

DEVICE_TOPO = Dict[int, Tuple[str, int]]


def create_device_topo(
    device_names: List[str], device_cnts: List[int], use_tp: bool = False
) -> DEVICE_TOPO:
    device_topo = {}
    rank = 0
    for device_name, device_cnt in zip(device_names, device_cnts):
        if not use_tp:
            for i in range(device_cnt):
                device_topo[rank + i] = (device_name, 1)
                rank += device_cnt
        else:
            # tp in the same device
            device_topo[rank] = (device_name, device_cnt)
            rank += 1
    return device_topo
