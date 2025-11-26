#!/bin/bash
# GPU Memory Cleanup Script for Jetson Orin Nano
# Run this before starting Flask if you encounter CUDA memory errors

echo "🧹 Cleaning up GPU memory on Jetson Orin Nano..."

# Kill any zombie Python processes
echo "1️⃣  Checking for zombie Python processes..."
PYTHON_PIDS=$(pgrep -f python)
if [ -n "$PYTHON_PIDS" ]; then
    echo "   Found Python processes: $PYTHON_PIDS"
    echo "   Killing old Python processes..."
    sudo pkill -9 -f python
    sleep 2
else
    echo "   ✓ No zombie Python processes found"
fi

# Clear system cache
echo ""
echo "2️⃣  Clearing system cache..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
echo "   ✓ System cache cleared"

# Check GPU memory status
echo ""
echo "3️⃣  Checking GPU memory status..."
if command -v tegrastats &> /dev/null; then
    echo "   Running tegrastats for 2 seconds..."
    timeout 2 tegrastats 2>/dev/null || true
fi

# Check CUDA availability
echo ""
echo "4️⃣  Verifying CUDA..."
python3 -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'   ✓ CUDA available: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024**3)
    print(f'   ✓ Total GPU memory: {total_mem:.2f} GB')
    allocated = torch.cuda.memory_allocated(0) / (1024**2)
    cached = torch.cuda.memory_reserved(0) / (1024**2)
    print(f'   ✓ Currently allocated: {allocated:.2f} MB')
    print(f'   ✓ Currently cached: {cached:.2f} MB')
else:
    print('   ⚠️  CUDA not available')
" 2>/dev/null

echo ""
echo "✅ Cleanup complete! You can now start Flask:"
echo "   python3 run.py"
