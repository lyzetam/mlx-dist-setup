
import mlx.core as mx
import socket

# Initialize distributed
world = mx.distributed.init()
rank = world.rank()
size = world.size()
hostname = socket.gethostname()

print(f"Rank {rank}/{size} on {hostname}")

# Set GPU
mx.set_default_device(mx.gpu)
print(f"  GPU: {mx.metal.is_available()}")
print(f"  Device: {mx.default_device()}")

# Test distributed computation
if size > 1:
    local_value = mx.array([float(rank)])
    print(f"  Local value: {local_value.item()}")

    # All-reduce sum
    sum_value = mx.distributed.all_sum(local_value)
    mx.eval(sum_value)

    if rank == 0:
        expected = sum(range(size))
        print(f"\nAll-reduce sum: {sum_value.item()} (expected: {expected})")
        success = abs(sum_value.item() - expected) < 0.001
        print(f"{'✓' if success else '✗'} Test {'PASSED' if success else 'FAILED'}!")
else:
    print("\n⚠️  Only 1 process - need multiple processes to test distributed ops")
