#!/bin/bash
# Quick connectivity test

echo "🔍 Testing basic connectivity..."

echo "1. Ping test:"
ping -c 1 mm1.local && echo "✅ mm1.local ping OK" || echo "❌ mm1.local ping failed"
ping -c 1 mm2.local && echo "✅ mm2.local ping OK" || echo "❌ mm2.local ping failed"

echo ""
echo "2. SSH test (simple):"
ssh -o ConnectTimeout=3 mm@mm1.local "hostname" && echo "✅ mm1.local SSH OK" || echo "❌ mm1.local SSH failed"
ssh -o ConnectTimeout=3 mm@mm2.local "hostname" && echo "✅ mm2.local SSH OK" || echo "❌ mm2.local SSH failed"

echo ""
echo "3. MLX test on remote nodes:"
ssh mm@mm1.local "python3 -c 'import mlx.core as mx; print(f\"mm1 MLX: {mx.metal.is_available()}\")'" 2>/dev/null && echo "✅ mm1 MLX OK" || echo "❌ mm1 MLX failed"
ssh mm@mm2.local "python3 -c 'import mlx.core as mx; print(f\"mm2 MLX: {mx.metal.is_available()}\")'" 2>/dev/null && echo "✅ mm2 MLX OK" || echo "❌ mm2 MLX failed"
