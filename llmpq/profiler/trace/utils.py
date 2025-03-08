import json 
from typing import List, Dict, Any
from collections import defaultdict
# perform the analyze on the trace 


def sort_event_by_ts(trace_event:List[Dict[str, Any]]):
    return sorted(trace_event, key=lambda x: x['ts'])

def get_cat_events(cat: str, trace_event:List[Dict[str, Any]]):
    events = [event for event in trace_event if 'cat' in event and event['cat'] == cat]
    return sort_event_by_ts(events)

def get_module_events(module_names: str, trace_event:List[Dict[str, Any]]):
    python_function_events = get_cat_events('python_function', trace_event)
    module_event_dict = {}
    for module_name in module_names:
        events = [event for event in python_function_events if module_name in event['name']]
        module_event_dict[module_name] = events
    return module_event_dict
