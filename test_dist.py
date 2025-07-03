
import mlx.core as mx

# This should now work properly with MPI
world = mx.distributed.init()
rank = world.rank()
size = world.size()

print(f"Rank {rank}/{size}: MLX distributed initialized!")

# Set GPU
mx.set_default_device(mx.gpu)
print(f"Rank {rank}: GPU = {mx.metal.is_available()}")

# Test communication
if size > 1:
    local = mx.array([float(rank)])
    total = mx.distributed.all_sum(local)
    mx.eval(total)

    if rank == 0:
        print(f"\nDistributed sum: {total.item()} (expected: {sum(range(size))})")
        print("✓ Success!")
