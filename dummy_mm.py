import torch
import time

nrank = torch.cuda.device_count()
shape = (8192, 8192)
lhs = [torch.randn(shape, device=f"cuda:{rank}") for rank in range(nrank)]
rhs = [torch.randn(shape, device=f"cuda:{rank}") for rank in range(nrank)]

while True:
    for i in range(1, nrank):
        o = lhs[i] @ rhs[i]
    time.sleep(0.005)


# expect rpdb==0.1.6
def debug_rpdb():
    import rpdb
    import time
    try:
        rpdb.Rpdb(port=6678 + int(time.time()) % 6677).set_trace()
    except Exception as ee:
        print(f"luqi failed: {ee}")
        while True:
            time.sleep(1000)