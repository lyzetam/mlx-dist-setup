
import mlx.core as mx
import socket
import sys

# Initialize distributed
world = mx.distributed.init()
rank = world.rank()
hostname = socket.gethostname()

# Set GPU and test
mx.set_default_device(mx.gpu)

# Simple GPU test
try:
    test_array = mx.ones((100, 100))
    mx.eval(test_array)
    gpu_status = "✅ GPU_WORKING"
except Exception as e:
    gpu_status = f"❌ GPU_ERROR: {e}"

# Print in a format easy to parse
print(f"RANK_{rank}|HOST_{hostname}|{gpu_status}")

# Sync
mx.eval(mx.distributed.all_sum(mx.array([1.0])))
