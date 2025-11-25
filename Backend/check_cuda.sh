#!/bin/bash

# Enable CUDA on Jetson Orin Nano
# Run this if you're getting "Using CPU mode" warning

echo "🔍 Checking CUDA availability..."
echo "================================"

# Check NVIDIA driver
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ nvidia-smi not working! CUDA driver issue."
    echo "Try: sudo systemctl restart nvargus-daemon"
    exit 1
fi

echo ""
echo "🐍 Checking PyTorch CUDA..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("✅ CUDA is working!")
else:
    print("❌ CUDA not available in PyTorch")
    print("\nYou need to install PyTorch with CUDA support for Jetson!")
EOF

echo ""
echo "📦 Current torch version:"
pip3 show torch | grep Version

echo ""
echo "💡 To enable CUDA:"
echo "================================"
echo "1. Check JetPack version:"
echo "   cat /etc/nv_tegra_release"
echo ""
echo "2. For JetPack 5.x (L4T 35.x):"
echo "   Download from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "   Or use:"
echo "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "3. For JetPack 6.x (L4T 36.x):"
echo "   Download from NVIDIA forums"
echo ""
echo "4. Verify installation:"
echo "   python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "5. Restart Flask:"
echo "   python3 run.py"
