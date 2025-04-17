import numpy as np 
from transformers import OPTConfig
from llmpq.utils import convert_to_unit
from llmpq.config import PQConfig
from llmpq.utils.v1.device import get_single_device_mem_constraints

class ModelMemEstimator:
    def __init__(self, h1, h2, b, s, n, vocab_size=None, max_position_embeddings=None, word_embed_proj_dim=None) -> None:
        # Refer to the flexGEN
        # h1 hidden dimension
        # h2 hidden dimension of second mlp
        # b batch size
        # s sequence length
        # n generated token numbers
        self.h1 = h1
        self.h2 = h2
        self.b = b
        self.s = s
        self.n = n
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.word_embed_proj_dim = word_embed_proj_dim
    
    def calculate_prepost_mem(self, unit='b', bit=16):
        # contain token embedding and positional embedding. Positiona
        if self.vocab_size is None:
            print("Token embedding dim is not specified")
            return 0
        # import pdb; pdb.set_trace()
        # calculate each embedding size
        # 32 size
        token_embedding_size = self.vocab_size * self.word_embed_proj_dim * 4
        max_pos_embedding_size = self.max_position_embeddings * self.h1 * 4
        # there exists a project_out / project_in for the max_pos_embedding if work_embed_proj_dim != h1
        if self.word_embed_proj_dim != self.h1:
            max_pos_embedding_size += 2 * self.h1 * self.word_embed_proj_dim * 4
        # there could be project_in and out here.
        # lm_head
        lm_head_weight_size = self.vocab_size * self.word_embed_proj_dim * 4
        mem_b = token_embedding_size + max_pos_embedding_size + lm_head_weight_size
        mem_b = mem_b * bit / 32
        mem_b += self.calculate_single_layer_ln_weight() * bit / 16
        return convert_to_unit(mem_b, unit), f"{convert_to_unit(mem_b, unit)} {unit}"
    
    def calculate_single_selfattn_mem(self):
        # QKV storage + OUT projection, 4 linear
        # return bytes
        weight_size = 4 * self.h1 * self.h1 * 4 # 4 means fp32 storage weight has 4 bytes
        return weight_size 

    def calculate_single_FFN_mem(self):
        # 2 linear
        weight_size = 2 * self.h1 * self.h2 * 4
        return weight_size

    def calculate_single_decoder_layer_mem(self):
        # MHA + FFN + 2 linear + 2 layernorm
        # return bytes
        return self.calculate_single_selfattn_mem() + self.calculate_single_FFN_mem() 
    
    def calculate_single_layer_maximum_kv_cache(self):
        # print(self.b, self.h1, (self.s + self.n))
        size = self.b * self.h1 * (self.s + self.n) * 2 * 2 #(k and v), store in fp16
        return size 
    
    def calculate_single_layer_ln_weight(self):
        size = self.h1 * 2 * 2 # 2 means fp16
        return size
    
    def calculate_multiple_layer_selfattn_mem(self, layer_num):
        return self.calculate_single_selfattn_mem() * layer_num

    def calculate_multiple_layer_FFN_mem(self, layer_num):
        return self.calculate_single_FFN_mem() * layer_num
    
    def calculate_multiple_layer_decoder_layer_mem(self, layer_num):
        return self.calculate_single_decoder_layer_mem() * layer_num
    
    def calculate_multiple_layer_kv_cache(self, layer_num):
        return self.calculate_single_layer_maximum_kv_cache() * layer_num
    
    def calculate_kv_occupation_of_partition(self, partition, unit='b'):
        # partition should be with format
        # {0: {"shard": [0,1], "bits": [8,8]}}
        all_size_estimation = 0
        for layer, config in partition.items():
            shard = config["shard"]
            bits = config["bits"]
            if len(shard) != len(bits):
                raise ValueError("shard and bits should have same length")

            for idx, value in enumerate(shard):
                if value == 0:
                    # add the kv size
                    kv_size = self.calculate_single_layer_maximum_kv_cache() # calculate in fp16
                    # bits
                    bit = bits[idx]
                    if bit == '8:tc': # only for tensorcore, we store the kv in INT8
                        bit = 8
                        kv_size = kv_size * bit / 16 # the feature supported by pure int8
                    all_size_estimation += kv_size 
        return convert_to_unit(all_size_estimation, unit), f'{convert_to_unit(all_size_estimation, unit)} {unit}'
    
    def calculate_temp_embedding_tensor_size(self, unit='b'):
        all = self.b * self.s * (3 * self.h1 + 2 * self.word_embed_proj_dim)
        return convert_to_unit(all, unit), f'{convert_to_unit(all, unit)} {unit}'
    
    def calculate_temp_tensor_size_prefill(self, unit='b'):
        # a layer norm
        attn_ln_tmp_size = self.b * self.s * self.h1 * 2 # by default to 16
        # 3QKV + 1 proj
        qkv_tmp_size = 4 * self.b * self.s * self.h1 * 2 # by default to 16
        # softmax, 32 bit
        softmax_tmp_size = self.b * self.s * self.h1 * 4 # by default to 16
        # 2 BMM (for qk_bmm, there is a softmax)
        bmm_tmp_size = (self.b * self.s * self.s + self.b * self.s * self.h1) * 2 # by default to 16
        # tmp buffer for kv cache
        kv_cache_tmp = 0
        # ffn
        # a layer norm
        ffn_ln_tmp_size = self.b * self.s * self.h1 * 2 # by default to 16
        # activation
        activation_tmp_size = self.b * self.s * self.h2 * 2 # by default to 16
        # fc1 and fc2
        fc_tmp_size = self.b * self.s * (self.h1 + self.h2) * 2 # by default to 16
        # total
        total_tmp_size = attn_ln_tmp_size + qkv_tmp_size + bmm_tmp_size + kv_cache_tmp + softmax_tmp_size + \
              ffn_ln_tmp_size + activation_tmp_size + fc_tmp_size 
        return convert_to_unit(total_tmp_size, unit), f'{convert_to_unit(total_tmp_size, unit)} {unit}'
    
    def calculate_temp_tensor_size_next_i(self, unit='b'):
        # attn
        # a layer norm
        attn_ln_tmp_size = self.b * self.h1 * 2 # by default to 16
        # 3QKV + 1 proj
        qkv_tmp_size = 4 * self.b * self.h1 * 2 # by default to 16
        # 2 BMM (for qk_bmm, there is a softmax)
        bmm_tmp_size = (self.b * (self.s + self.n) + self.b * self.h1) * 2 # by default to 16
        # 32
        softmax_tmp_size = self.b * (self.s + self.n) * 4
        # tmp buffer for kv cache
        kv_cache_tmp = 2 * (self.b * (self.s + self.n) * self.h1) * 2 # by default to 16
        # ffn
        # a layer norm
        ffn_ln_tmp_size = self.b * self.h1 * 2 # by default to 16
        # activation
        activation_tmp_size = self.b * self.h2 * 2 # by default to 16
        # fc1 and fc2
        fc_tmp_size = self.b * (self.h1 + self.h2) * 2 # by default to 16
        # total
        total_tmp_size = attn_ln_tmp_size + qkv_tmp_size + bmm_tmp_size + kv_cache_tmp + softmax_tmp_size + \
              ffn_ln_tmp_size + activation_tmp_size + fc_tmp_size 
        return convert_to_unit(total_tmp_size, unit), f'{convert_to_unit(total_tmp_size, unit)} {unit}'

    # return in bytes
    def calculate_temp_tensor_size(self, unit='b'):
        max_temp = max(self.calculate_temp_tensor_size_prefill(unit)[0], \
                       self.calculate_temp_tensor_size_next_i(unit)[0], \
                        self.calculate_temp_embedding_tensor_size(unit)[0])
        return max_temp, f'{max_temp} {unit}'
    
    def calculate_temp_tensor_size_with_bz(self, bz_prefill, bz_decode, unit='b'):
        ratio_decode = bz_decode / self.b
        ratio_prefill = bz_prefill / self.b
        # calculate the temp tensor size with different batch size
        # return in bytes
        max_temp = max(self.calculate_temp_tensor_size_prefill(unit)[0] * ratio_prefill, \
                       self.calculate_temp_tensor_size_next_i(unit)[0] * ratio_decode, \
                        self.calculate_temp_embedding_tensor_size(unit)[0] * ratio_prefill)
        return max_temp, f'{max_temp} {unit}'

    def calculate_model_occupation_of_partition(self, partition, unit='b'):
        # partition should be with format
        # {0: {"shard": [0,1], "bits": [8,8]}}
        all_size_estimation = 0
        for layer, config in partition.items():
            shard = config["shard"]
            bits = config["bits"]
            
            if len(shard) != len(bits):
                raise ValueError("shard and bits should have same length")

            for idx, value in enumerate(shard):
                if value == 0:
                    selfattn_mem = self.calculate_single_selfattn_mem()
                    # bits
                    bit = bits[idx]
                    if type(bit) != int:
                        bit = 8
                    selfattn_mem = selfattn_mem * bit / 32 
                    ln_size = self.calculate_single_layer_ln_weight()
                    all_size_estimation += selfattn_mem + ln_size
                elif value == 1:
                    ffn_mem = self.calculate_single_FFN_mem()
                    bit = bits[idx]
                    if type(bit) != int:
                        bit = 8
                    ffn_mem = ffn_mem * bit / 32
                    all_size_estimation += ffn_mem
        return convert_to_unit(all_size_estimation, unit), f'{convert_to_unit(all_size_estimation, unit)} {unit}'

    

    def calculate_maximum_mem_occupation_of_partition(self, partition, unit='b'):
        # partition should be with format
        # {0: {"shard": [0,1], "bits": [8,8]}}
        all_size_estimation = 0
        kv_mem = self.calculate_kv_occupation_of_partition(partition, unit)[0]
        model_mem = self.calculate_model_occupation_of_partition(partition, unit)[0]
        all_size_estimation = kv_mem + model_mem
        return all_size_estimation, f"{all_size_estimation} {unit}" 
    
    def estimate_hidden_space(self):
        print(self.b, self.s + self.n - 1, self.h1)
        return self.h1 * self.b * (self.s + self.n - 1)

    def estimate_single_layer_kv_cache(self, unit='b'):
        print(self.b, (self.s + self.n - 1), self.h1)
        return self.calculate_single_layer_maximum_kv_cache(), f"{convert_to_unit(self.calculate_single_layer_maximum_kv_cache(), unit)} {unit}"


def estimate_single_layer_mem(estimator, shard, bit):
    partition = {0: {"shard": [shard], "bits": [bit]}}
    mem_require, _ = estimator.calculate_maximum_mem_occupation_of_partition(partition, unit=PQConfig.MEM_UNIT)
    return mem_require

def estimate_all_layer_mem(estimator, layers, bit_map):
    all_mem_require = 0
    for idx, shard in enumerate(layers):
        bit = bit_map[idx]
        mem_require = estimate_single_layer_mem(estimator, shard, bit)
        all_mem_require += mem_require
    return all_mem_require

def get_mem_with_layer_bit_pair(bit_pairs, model_mem_estimator): 
    mem_bits_vector = np.zeros(len(bit_pairs))
    for idx, bit_pair in enumerate(bit_pairs):
        attn_bit, ffn_bit = bit_pair
        attn_mem = estimate_single_layer_mem(model_mem_estimator, 0, attn_bit)
        ffn_mem = estimate_single_layer_mem(model_mem_estimator, 1, ffn_bit)
        mem = attn_mem + ffn_mem
        mem_bits_vector[idx] = mem
    return mem_bits_vector

def create_mem_estimator(h1, h2, b, s, n, config):
    vocab_size = config.vocab_size
    if isinstance(config, OPTConfig):
        max_position_embeddings = config.max_position_embeddings
        word_embed_proj_dim = config.word_embed_proj_dim
    else:
        max_position_embeddings = 0
        word_embed_proj_dim = h1
    model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n, \
                                            vocab_size=vocab_size, max_position_embeddings=max_position_embeddings, word_embed_proj_dim=word_embed_proj_dim)
    return model_mem_estimator


def get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size):
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
    M = np.tile(mem_bits_vector, (group_L, 1)) * group_size # repeat the mem_bits_vector for group_L times
    M = np.ceil(M).astype(int) # ceil
    return M

def get_device_topo_available_mem_with_order(current_D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=1):
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in current_D.items()]) 
    # reduce the embedding size on device 0
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    temp_later_decode = model_mem_estimator.calculate_temp_tensor_size_next_i(unit='MB')[0]
    M_d[0] -= post_pre_mem
    if len(M_d) > 1:
        M_d[1:] -= temp_later_decode * time_mult_times
    M_d[0] -= max(temp_tensor_mem, temp_later_decode * time_mult_times)
    return M_d


def estimate_single_device_mem(layers_range, bit_assignment, model_mem_estimator):
    i, j = layers_range
    i_to_j_mem = 0
    # k % 2 means shard
    for k in range(i, j):
        layer_x_mem = estimate_single_layer_mem(model_mem_estimator, 0, bit_assignment[k * 2]) + \
                        estimate_single_layer_mem(model_mem_estimator, 1, bit_assignment[k * 2 + 1])
        # print("layer_x_mem", layer_x_mem)
        i_to_j_mem += layer_x_mem
    return i_to_j_mem


# first make sure the partition is within the memory budget
def check_memory_budget_single_device(device_mem, device_rank, layers_range, bit_assignment, model_mem_estimator):
    i_to_j_mem = estimate_single_device_mem(layers_range, bit_assignment, model_mem_estimator)
    if i_to_j_mem > device_mem:
        print(f"memory budget exceeded for device {device_rank}, {i_to_j_mem} > {device_mem}")
        return False
    return True

def check_memory_budget(res, model_mem_estimator, name='llm_pq'):
    plan = res['plan']
    partition_result = plan['partition_result']
    bit_assignment = plan['bit_assignment']
    D = res['D']
    prefill_bz = res['prefill_bz']
    bz_decode_max = res['bz_decode_max']
    # print("verify memory budget for", name)
    D_mem = get_device_topo_available_mem_with_order(D, model_mem_estimator, prefill_bz, bz_decode_max)
    for device_rank, layers_range in partition_result.items():
        device_mem = D_mem[device_rank]
        flag = check_memory_budget_single_device(device_mem, device_rank, layers_range, bit_assignment, \
                                           model_mem_estimator)
        if not flag:
            print("memory budget exceeded, return False", name)
            import pdb; pdb.set_trace()
            return False
    # print("all passed")
    return True