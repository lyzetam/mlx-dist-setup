
import mlx.core as mx
import socket
import os

def test_cluster():
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()
    hostname = socket.gethostname()

    mx.set_default_device(mx.gpu)

    print(f"[{rank}/{size}] {hostname} - GPU: {mx.metal.is_available()}")

    # Test communication
    test_data = mx.array([float(rank)])
    result = mx.distributed.all_sum(test_data)
    mx.eval(result)

    if rank == 0:
        expected = sum(range(size))
        print(f"\nCluster health: {'✅ GOOD' if abs(result.item() - expected) < 0.001 else '❌ FAILED'}")
        print(f"All-reduce test: {result.item()} (expected: {expected})")

if __name__ == "__main__":
    test_cluster()
