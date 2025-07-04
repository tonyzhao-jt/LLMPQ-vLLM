import json
from typing import Dict, Any
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .utils import get_cat_events, get_module_events


def parse_module_avg_cost(
    module_cnt: Dict[str, int], trace_path: str
) -> Dict[str, Any]:
    with open(trace_path, "r") as f:
        trace = json.load(f)

    device_properties = trace["deviceProperties"]
    gpu_name = device_properties[0]["name"]
    trace_event = trace["traceEvents"]

    modules_names = list(module_cnt.keys())

    def filter_cuda_launch_kernel():
        cuda_runtime_events = get_cat_events("cuda_runtime", trace_event)
        with tqdm(cuda_runtime_events, desc="Filtering cudaLaunchKernel") as pbar:
            return [event for event in pbar if event["name"] == "cudaLaunchKernel"]

    def filter_module_events():
        return get_module_events(modules_names, trace_event)

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(filter_cuda_launch_kernel)
        future2 = executor.submit(filter_module_events)
        cudaLaunchKernelEvents = future1.result()
        module_event_dict = future2.result()

    def match_kernel_ranges():
        module_event_to_cudaLaunchKernelEvents = {}
        total_events = sum(len(events) for events in module_event_dict.values())
        with tqdm(
            total=total_events, desc="Matching kernel ranges", unit="module_event"
        ) as pbar:
            for module_name, events in module_event_dict.items():
                for e_id, event in enumerate(events):
                    e_name = f"{module_name}_{e_id}"
                    ts = event["ts"]
                    dur = event["dur"]
                    start, end = ts, ts + dur
                    tgt_events = [
                        e for e in cudaLaunchKernelEvents if start <= e["ts"] <= end
                    ]
                    module_event_to_cudaLaunchKernelEvents[e_name] = tgt_events
                    pbar.update()
        return module_event_to_cudaLaunchKernelEvents

    def process_kernels(module_event_to_cudaLaunchKernelEvents):
        kernel_events = get_cat_events("kernel", trace_event)
        module_cuda_kernel_dur = defaultdict(list)
        total_kernels = sum(
            len(v) for v in module_event_to_cudaLaunchKernelEvents.values()
        )
        with tqdm(
            total=total_kernels, desc="Processing kernels", unit="kernel"
        ) as main_pbar:
            for (
                module_name,
                cuda_events,
            ) in module_event_to_cudaLaunchKernelEvents.items():
                with tqdm(
                    cuda_events, desc=f"Matching {module_name}", leave=False
                ) as sub_pbar:
                    for cuda_event in sub_pbar:
                        matched = []
                        find_one = False
                        with tqdm(
                            kernel_events, desc="Scanning kernels", leave=False
                        ) as kernel_pbar:
                            for event in kernel_pbar:
                                try:
                                    if (
                                        event["args"]["correlation"]
                                        == cuda_event["args"]["correlation"]
                                    ):
                                        find_one = True
                                        matched.append(event)
                                    elif find_one:
                                        break
                                except KeyError:
                                    continue
                        if matched:
                            total_dur = sum(m["dur"] for m in matched)
                            module_cuda_kernel_dur[module_name].append(total_dur)
                        main_pbar.update(1)
        return module_cuda_kernel_dur

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(match_kernel_ranges)
        module_event_to_cudaLaunchKernelEvents = future1.result()
        future2 = executor.submit(
            process_kernels, module_event_to_cudaLaunchKernelEvents
        )
        module_cuda_kernel_dur = future2.result()

    def process_run_engine_events():
        run_engine_events = get_module_events(["run_engine"], trace_event)["run_engine"]
        run_e_name_mapping = defaultdict(list)
        with tqdm(
            enumerate(run_engine_events),
            total=len(run_engine_events),
            desc="Processing runs",
        ) as pbar:
            for run_id, engine_event in pbar:
                run_start = engine_event["ts"]
                run_end = run_start + engine_event["dur"]
                for module_name, events in module_event_dict.items():
                    for e_id, event in enumerate(events):
                        e_name = f"{module_name}_{e_id}"
                        ts = event["ts"]
                        dur = event["dur"]
                        if ts >= run_start and (ts + dur) <= run_end:
                            run_e_name_mapping[run_id].append(e_name)
        return run_e_name_mapping

    def classify_phases(run_e_name_mapping):
        run_e_name_pd_mapping = defaultdict(dict)
        with tqdm(run_e_name_mapping.items(), desc="Classifying phases") as pbar:
            for run_id, e_names in pbar:
                prefill = []
                decode = []
                for module in modules_names:
                    module_events = [e for e in e_names if module in e]
                    cnt = module_cnt[module]
                    if not module_events:
                        continue
                    prefill.extend(module_events[:cnt])
                    decode.extend(module_events[cnt:])
                run_e_name_pd_mapping[run_id] = {"prefill": prefill, "decode": decode}
        return run_e_name_pd_mapping

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(process_run_engine_events)
        run_e_name_mapping = future1.result()
        future2 = executor.submit(classify_phases, run_e_name_mapping)
        run_e_name_pd_mapping = future2.result()

    def aggregate_results(run_e_name_pd_mapping):
        cost_table = defaultdict(dict)
        total_ops = sum(
            len(pd) for run in run_e_name_pd_mapping.values() for pd in run.values()
        )
        with tqdm(total=total_ops, desc="Aggregating results") as main_pbar:
            for run_id, pd_mapping in run_e_name_pd_mapping.items():
                with tqdm(
                    pd_mapping.items(), desc=f"Run {run_id}", leave=False
                ) as sub_pbar:
                    for phase, e_names in sub_pbar:
                        for e_name in e_names:
                            module = next(
                                (m for m in modules_names if m in e_name), None
                            )
                            assert module, f"Module not found in {e_name}"
                            if phase not in cost_table[module]:
                                cost_table[module][phase] = []
                            cost_table[module][phase].extend(
                                module_cuda_kernel_dur[e_name]
                            )
                            main_pbar.update(1)
        return cost_table

    def calculate_averages(cost_table):
        with tqdm(cost_table.items(), desc="Calculating averages") as pbar:
            for module, phases in pbar:
                for phase in phases:
                    values = phases[phase]
                    phases[phase] = sum(values) / len(values) if values else 0.0
        return cost_table

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(aggregate_results, run_e_name_pd_mapping)
        cost_table = future1.result()
        future2 = executor.submit(calculate_averages, cost_table)
        cost_table = future2.result()

    return cost_table, gpu_name
