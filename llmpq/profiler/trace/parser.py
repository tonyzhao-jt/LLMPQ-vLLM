
import json
from typing import Dict, Any
from collections import defaultdict
from .utils import get_cat_events, get_module_events

def parse_module_avg_cost(module_cnt: Dict[str, int], trace_path: str) -> Dict[str, Any]:

    with open(trace_path, 'r') as f:
        trace = json.load(f)

    device_properties = trace['deviceProperties']
    gpu_name = device_properties[0]['name']
    trace_event = trace['traceEvents']


    modules_names = list(module_cnt.keys())
    # filter cuda lanch kernel
    cuda_runtime_events = get_cat_events('cuda_runtime', trace_event)
    cudaLaunchKernelEvents = [event for event in cuda_runtime_events if event['name'] == 'cudaLaunchKernel']

    # filter the layer name 
    module_event_dict = get_module_events(modules_names, trace_event)

    # get the start time and end of each module, assign the cudaLaunchKernelEvents to the module
    module_event_to_cudaLaunchKernelEvents = {}
    for module_name, events in module_event_dict.items():
        for e_id, event in enumerate(events):
            e_name = f"{module_name}_{e_id}"
            ts = event['ts']
            dur = event['dur']
            start, end = ts, ts + dur
            # filter the cudaLuanchKernelEvents to find the event that lies in the range of start and end
            tgt_cudaLaunchKernelEvents = [event for event in cudaLaunchKernelEvents if event['ts'] >= start and event['ts'] <= end]
            module_event_to_cudaLaunchKernelEvents[e_name] = tgt_cudaLaunchKernelEvents

    # get all kernels
    kernel_events = get_cat_events('kernel', trace_event)
    # map the cudalaunch kernel events to the kernel dur
    module_cuda_kernel_dur = defaultdict(list)
    for module_name, cudaLaunchKernelEvents in module_event_to_cudaLaunchKernelEvents.items():
        for cudaLaunchKernelEvent in cudaLaunchKernelEvents:
            # find the kernel event that has the same args
            kernel_event = [event for event in kernel_events if event['args']['External id'] == cudaLaunchKernelEvent['args']['External id']]
            if len(kernel_event) == 0:
                continue
            kernel_event = kernel_event[0]
            kernel_dur = kernel_event['dur']
            module_cuda_kernel_dur[module_name].append(kernel_dur)

    # now the module_cuda_kernel_dur is like 'LogitsProcessor_175': [8.384, 301.855], 'LogitsProcessor_176': [8.417, 300.992]
    # we can count the latency now. 
    run_engine_events = get_module_events(['run_engine'], trace_event)['run_engine']
    run_e_name_mapping = defaultdict(list)
    for run_id, engine_event in enumerate(run_engine_events):
        run_start = engine_event['ts']
        run_end = run_start + engine_event['dur']
        for module_name, events in module_event_dict.items():
            for e_id, event in enumerate(events):
                e_name = f"{module_name}_{e_id}"
                ts = event['ts']
                dur = event['dur']
                start, end = ts, ts + dur
                if start >= run_start and end <= run_end:
                    run_e_name_mapping[run_id].append(e_name)

    run_e_name_pd_mapping = defaultdict(dict)

    # for each run, the first one of each time is the prefill time, and the remaining is the decode time
    for run_id, e_names in run_e_name_mapping.items():
        prefill_enames = []
        decode_enames = []
        for module_name in modules_names:
            module_events = [event for event in e_names if module_name in event]
            cnt = module_cnt[module_name]
            if len(module_events) == 0:
                continue
            prefill_enames.extend(module_events[:cnt])
            decode_enames.extend(module_events[cnt:])
        
        run_e_name_pd_mapping[run_id]['prefill'] = prefill_enames
        run_e_name_pd_mapping[run_id]['decode'] = decode_enames
        

    # final agg result 
    # cost: prefill, module -> avg cost 
    cost_table = defaultdict(dict)
    for run_id, e_name_pd_mapping in run_e_name_pd_mapping.items():
        for pd, e_names in e_name_pd_mapping.items():
            for e_name in e_names:
                e_name_id = None 
                for module_name in modules_names:
                    if module_name in e_name:
                        e_name_id = module_name
                        break
                assert e_name_id is not None, f"e_name_id is None, e_name: {e_name}"
                if e_name_id not in cost_table:
                    cost_table[e_name_id] = {}
                if pd not in cost_table[e_name_id]:
                    cost_table[e_name_id][pd] = []
                cost_table[e_name_id][pd].extend(module_cuda_kernel_dur[e_name])

    # perform avg
    for e_name, pd_cost in cost_table.items():
        for pd, cost in pd_cost.items():
            cost_table[e_name][pd] = sum(cost) / len(cost)
    
    return cost_table, gpu_name