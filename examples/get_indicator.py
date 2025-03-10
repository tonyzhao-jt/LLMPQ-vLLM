from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
from time import perf_counter
import torch 
from collections import defaultdict
import numpy as np 
from llmpq.profiler.indicator import get_loaders
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model_cfg = AutoConfig.from_pretrained(model_id)
num_layers = model_cfg.num_hidden_layers
c4_loader = get_loaders('c4', model=model_id)
import pdb; pdb.set_trace()


# create torch dataloader
from torch.utils.data import DataLoader
def tokenize_function(examples):
    return tokenizer(examples["text"])


@torch.no_grad()
def bloom_profile(self, dataloader):
    """Optimized model profiling function with reduced device transfers and improved memory management."""
    print('Starting profile...')
    profile_start_time = perf_counter()
    
    # Device context manager
    def device_manager(module, device):
        """Context manager for device transition"""
        original_devices = [p.device for p in module.parameters()]
        yield module.to(device)
        for p, original_device in zip(module.parameters(), original_devices):
            p.data = p.to(original_device)

    # 配置初始化
    model = self.model
    device = self.device
    model.config.use_cache = False
    layers = model.transformer.h

    # 初始化统计数据结构
    collected_info = defaultdict(lambda: {
        'x_var': 0.0,
        'wmax': -np.inf,
        'wmin': np.inf
    })

    # 前向钩子收集器
    class StatsCollector:
        def __init__(self, layer_idx, name):
            self.layer_idx = layer_idx
            self.name = name

        def __call__(self, module, inp, out):
            # 权重统计
            weight = module.weight.data.float()
            collected_info[(self.layer_idx, self.name)]['wmax'] = weight.max(dim=-1, keepdim=True).values.cpu().numpy()
            collected_info[(self.layer_idx, self.name)]['wmin'] = weight.min(dim=-1, keepdim=True).values.cpu().numpy()
            
            # 输入统计
            x = inp[0].float()
            collected_info[(self.layer_idx, self.name)]['x_var'] += x.var(dim=-1, keepdim=True).cpu().numpy()

    # 主处理逻辑
    with torch.inference_mode(), device_manager(model.transformer, device) as model:
        # 初始化输入缓存
        inps = torch.zeros(
            (self.args.nsamples, self.seqlen, model.config.hidden_size),
            dtype=next(model.parameters()).dtype,
            device=device
        )
        
        # 输入捕获
        input_collector = []
        def input_hook(module, inp, out):
            nonlocal input_collector
            input_collector.append(inp[0])
        handle = layers[0].register_forward_hook(input_hook)
        
        # 运行初始前向传播
        for batch in dataloader:
            model(batch[0].to(device))
            if len(input_collector) >= self.args.nsamples:
                break
        handle.remove()
        
        # 处理输入数据
        inps = torch.stack(input_collector[:self.args.nsamples])

        # 逐层分析
        handles = []
        for layer_idx, layer in enumerate(layers):
            with device_manager(layer, device):
                # 注册统计钩子
                for name, submodule in find_layers(layer).items():
                    handles.append(submodule.register_forward_hook(
                        StatsCollector(layer_idx, name)
                    ))
                
                # 运行前向传播
                outputs = []
                for inp in inps:
                    outputs.append(layer(inp.unsqueeze(0))[0])
                inps = torch.cat(outputs, dim=0)
                
                # 移除钩子
                for h in handles:
                    h.remove()
                handles.clear()

    # 后处理统计值
    for key in collected_info:
        collected_info[key]['x_var'] /= self.args.nsamples

    # 保存结果
    model_name = self.args.model.replace('/', '_')
    file_path = f'{model_name}_{self.args.dataset}_stat.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump({
            'statistics': collected_info,
            'duration': perf_counter() - profile_start_time
        }, f)

    print(f'Profile completed in {perf_counter()-profile_start_time:.2f}s')
    return file_path