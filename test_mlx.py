
import mlx.core as mx
import socket

# Initialize distributed
world = mx.distributed.init()
rank = world.rank()
size = world.size()
hostname = socket.gethostname()

# Set GPU
mx.set_default_device(mx.gpu)

print(f"Rank {rank}/{size} on {hostname}")
print(f"  GPU available: {mx.metal.is_available()}")

# Test distributed computation
local_value = mx.array([float(rank)])
sum_value = mx.distributed.all_sum(local_value)
mx.eval(sum_value)

if rank == 0:
    expected = sum(range(size))
    print(f"\nDistributed sum: {sum_value.item()} (expected: {expected})")
    if abs(sum_value.item() - expected) < 0.001:
        print("✓ MLX distributed test PASSED!")
    else:
        print("✗ MLX distributed test FAILED!")
