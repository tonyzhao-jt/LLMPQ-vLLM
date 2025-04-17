from typing import Tuple

def is_partition_config_valid(
    partition_config: Tuple[str, int], num_layers: int
):  # noqa
    config_str, pipeline_parallel_size = partition_config

    try:
        partitions = list(map(int, config_str.split(",")))
    except ValueError:
        return False

    if len(partitions) != pipeline_parallel_size:
        return False

    if sum(partitions) != num_layers:
        return False

    return True
