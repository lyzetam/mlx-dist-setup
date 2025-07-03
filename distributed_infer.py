# import socket
import mlx.core as mx
from mlx_lm import load, generate


def main():
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()

    mx.set_default_device(mx.gpu)

    if rank == 0:
        print(f"Running on {size} processes")

    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    prompt = f"Hello from rank {rank}!"
    result = generate(model, tokenizer, prompt, max_tokens=20)

    print(f"[{rank}/{size} on {socket.gethostname()}] {result}")


if __name__ == "__main__":
    main()