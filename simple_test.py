
import socket
import sys
print(f"Node: {socket.gethostname()}, Python: {sys.executable}")
try:
    import mlx.core as mx
    print("MLX available!")
except ImportError:
    print("MLX not available")
