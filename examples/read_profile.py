path = '/opt/tiger/Saber/llm_pq_v2/examples/cost_table_dict_tc.pkl'
import pickle
with open(path, 'rb') as f:
    cost_table_dict = pickle.load(f)

import pdb; pdb.set_trace()