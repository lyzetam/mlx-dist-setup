
import mlx.core as mx
import socket
import time

def main():
    # Initialize distributed
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()
    hostname = socket.gethostname()

    # Set GPU
    mx.set_default_device(mx.gpu)

    print(f"🖥️  RANK {rank}: Running on {hostname}")

    # Check if GPU is available and get basic info
    try:
        # Simple GPU memory check
        mx.eval(mx.ones((1000, 1000)))  # Small GPU operation
        mem_info = mx.metal.get_memory_info()
        allocated = mem_info["allocated"] / 1024 / 1024  # Convert to MB
        print(f"🚀 RANK {rank}: GPU working! {allocated:.1f}MB allocated on {hostname}")
    except Exception as e:
        print(f"❌ RANK {rank}: GPU issue on {hostname}: {e}")

    # Sync all processes
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print("\n✅ All processes synchronized!")

if __name__ == "__main__":
    main()
