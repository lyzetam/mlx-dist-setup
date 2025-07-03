
import mlx.core as mx
import socket
import os

print(f"Process on {socket.gethostname()}")
print(f"PID: {os.getpid()}")

# Initialize distributed
world = mx.distributed.init()
rank = world.rank()
size = world.size()

print(f"Rank {rank}/{size} initialized")

# Set GPU
mx.set_default_device(mx.gpu)
print(f"Rank {rank}: GPU = {mx.metal.is_available()}")

# Simple test
if size > 1:
    data = mx.array([float(rank)])
    result = mx.distributed.all_sum(data)
    mx.eval(result)

    if rank == 0:
        print(f"\nAll-reduce sum: {result.item()} (expected: {sum(range(size))})")
