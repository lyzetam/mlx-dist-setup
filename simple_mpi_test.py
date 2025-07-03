#!/usr/bin/env python3

import os
import socket
import sys

def main():
    print(f"[SIMPLE] Process starting...")
    print(f"[SIMPLE] Hostname: {socket.gethostname()}")
    print(f"[SIMPLE] PID: {os.getpid()}")
    print(f"[SIMPLE] Python: {sys.executable}")
    
    # Try to import MLX
    try:
        import mlx.core as mx
        print(f"[SIMPLE] MLX imported successfully")
        print(f"[SIMPLE] Metal available: {mx.metal.is_available()}")
    except Exception as e:
        print(f"[SIMPLE] MLX import failed: {e}")
        return
    
    # Try distributed init
    try:
        print(f"[SIMPLE] Attempting mx.distributed.init()...")
        world = mx.distributed.init()
        rank = world.rank()
        size = world.size()
        print(f"[SIMPLE] SUCCESS! Rank {rank}/{size}")
        
        # Simple test
        test_val = mx.array([float(rank)])
        result = mx.distributed.all_sum(test_val)
        mx.eval(result)
        print(f"[SIMPLE] All-reduce result: {result.item()}")
        
    except Exception as e:
        print(f"[SIMPLE] Distributed init failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
