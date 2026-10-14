#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -I. $(pkg-config --cflags opencv4) *.cpp -o evaluate $(pkg-config --libs opencv4)
