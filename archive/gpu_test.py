
import mlx.core as mx
import socket

def main():
    world = mx.distributed.init()
    rank = world.rank()
    hostname = socket.gethostname()

    mx.set_default_device(mx.gpu)

    try:
        # Test GPU operation
        test_array = mx.ones((1000, 1000))
        mx.eval(test_array)

        # Get GPU memory info
        mem_info = mx.metal.get_memory_info()
        allocated = mem_info["allocated"] / 1024 / 1024  # MB

        print(f"GPU_STATUS|RANK_{rank}|HOST_{hostname}|MEMORY_{allocated:.1f}MB|STATUS_OK")
    except Exception as e:
        print(f"GPU_STATUS|RANK_{rank}|HOST_{hostname}|ERROR_{str(e)}")

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

if __name__ == "__main__":
    main()
