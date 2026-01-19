#!/bin/bash
# Script for testing converted models on MT6989 platform

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"

echo "=========================================="
echo "MT6989 Platform Test Script"
echo "=========================================="
echo ""
echo "This script is for testing converted models on MT6989 platform."
echo ""
echo "Prerequisites:"
echo "1. Models should be converted using convert.py scripts"
echo "2. MT6989 development board or device should be connected"
echo "3. Neuron SDK should be installed on the target device"
echo ""
echo "Test scripts should:"
echo "1. Load converted models using Neuron API"
echo "2. Run inference on test images"
echo "3. Measure performance metrics (FPS, latency, memory)"
echo "4. Compare accuracy with local test results"
echo ""
echo "Please refer to Neuron SDK documentation for platform-specific testing."
echo ""
echo "Example test locations:"
echo "  - EDSR: $RESULTS_DIR/edsr/converted/"
echo "  - RCAN: $RESULTS_DIR/rcan/converted/"
