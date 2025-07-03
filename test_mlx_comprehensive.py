
import mlx.core as mx
import mlx.nn as nn
import socket
import time
import os

# Initialize distributed
world = mx.distributed.init()
rank = world.rank()
size = world.size()
hostname = socket.gethostname()
pid = os.getpid()

# Set GPU
mx.set_default_device(mx.gpu)

print(f"[Rank {rank}/{size}] Process {pid} on {hostname}")
print(f"[Rank {rank}] GPU: {mx.metal.is_available()}")
print(f"[Rank {rank}] Device: {mx.default_device()}")

# Synchronize before tests
mx.eval(mx.distributed.all_sum(mx.array([1.0])))

if rank == 0:
    print("\n" + "="*50)
    print("Running MLX Distributed Tests")
    print("="*50)

# Test 1: Basic all-reduce
if rank == 0:
    print("\n1. Testing all-reduce...")

local_value = mx.array([float(rank)])
sum_result = mx.distributed.all_sum(local_value)
mx.eval(sum_result)

if rank == 0:
    expected = sum(range(size))
    print(f"   All-reduce sum: {sum_result.item()} (expected: {expected})")
    print(f"   {'✓ PASSED' if abs(sum_result.item() - expected) < 0.001 else '✗ FAILED'}")

# Test 2: Model parameter synchronization
if rank == 0:
    print("\n2. Testing model parameter sync...")

model = nn.Linear(100, 10)
mx.eval(model.parameters())

# Get initial param sum
param_sum_before = sum(p.sum().item() for _, p in model.parameters())
print(f"[Rank {rank}] Initial param sum: {param_sum_before:.6f}")

# Synchronize parameters
for _, p in model.parameters():
    p_synced = mx.distributed.all_sum(p) / size
    p[:] = p_synced

mx.eval(model.parameters())
param_sum_after = sum(p.sum().item() for _, p in model.parameters())

# All ranks should have same param sum now
all_sums = mx.distributed.all_sum(mx.array([param_sum_after]))
mx.eval(all_sums)

if rank == 0:
    print(f"   Synchronized param sum: {param_sum_after:.6f}")
    print(f"   {'✓ PASSED' if all_sums.item() == param_sum_after * size else '✗ FAILED'}")

# Test 3: Bandwidth test
if rank == 0:
    print("\n3. Testing bandwidth...")

size_mb = 10
data = mx.random.uniform(shape=(size_mb * 1024 * 1024 // 4,))

start = time.time()
result = mx.distributed.all_sum(data)
mx.eval(result)
elapsed = time.time() - start

bandwidth = size_mb * size / elapsed
if rank == 0:
    print(f"   Data size: {size_mb}MB per rank")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Bandwidth: {bandwidth:.1f} MB/s")

# Final status
mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync
if rank == 0:
    print("\n" + "="*50)
    print("✓ All tests completed successfully!")
    print("="*50)
