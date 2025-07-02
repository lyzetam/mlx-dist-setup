#!/usr/bin/env bash
set -e

echo "Setting iogpu.wired_limit_mb to 120000 MB..."
sudo sysctl iogpu.wired_limit_mb=120000

echo "Wired memory limit set to 120 GB!"
