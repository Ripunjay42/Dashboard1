#!/bin/bash

# Jetson Orin Nano Setup Script
# This script fixes common segmentation fault issues on Jetson devices

echo "🚀 Jetson Orin Nano Setup Script"
echo "=================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This doesn't appear to be a Jetson device"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "📦 Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    v4l-utils

echo ""
echo "📹 Step 2: Configuring camera permissions..."
# Add user to video group
sudo usermod -a -G video $USER

echo ""
echo "🔧 Step 3: Setting up swap space (prevents out-of-memory crashes)..."
# Check current swap
current_swap=$(free -h | grep Swap | awk '{print $2}')
echo "Current swap: $current_swap"

if [ "$current_swap" == "0B" ] || [ -z "$current_swap" ]; then
    echo "Creating 4GB swap file..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    
    # Make swap permanent
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "✓ Swap created and enabled"
else
    echo "✓ Swap already configured"
fi

echo ""
echo "⚡ Step 4: Setting Jetson to maximum performance mode..."
# Set to max performance (MAXN mode)
sudo nvpmodel -m 0
sudo jetson_clocks

echo ""
echo "🎥 Step 5: Testing cameras..."
echo "Available video devices:"
ls -la /dev/video*

echo ""
echo "Camera information:"
for i in 0 1 2 3; do
    if [ -e "/dev/video$i" ]; then
        echo "Camera $i:"
        v4l2-ctl --device=/dev/video$i --all 2>/dev/null | grep -E "Driver|Card type|Pixel Format" || echo "  Unable to query"
    fi
done

echo ""
echo "🧪 Step 6: Testing GStreamer..."
# Test GStreamer plugins
gst-inspect-1.0 nvarguscamerasrc > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ nvarguscamerasrc available (CSI camera support)"
else
    echo "⚠️  nvarguscamerasrc not available (CSI cameras won't work)"
fi

gst-inspect-1.0 v4l2src > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ v4l2src available (USB camera support)"
else
    echo "❌ v4l2src not available"
fi

echo ""
echo "🐍 Step 7: Python environment check..."
python3 --version
pip3 --version

echo ""
echo "📚 Step 8: Installing Python dependencies..."
cd "$(dirname "$0")"
if [ -f requirements.txt ]; then
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    echo "✓ Python dependencies installed"
else
    echo "⚠️  requirements.txt not found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "IMPORTANT NOTES:"
echo "================"
echo "1. Log out and log back in for video group permissions to take effect"
echo "2. The system is now in maximum performance mode (high power consumption)"
echo "3. To test the camera:"
echo "   gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=640,height=480' ! nvvidconv ! xvimagesink"
echo "4. To run the Flask backend:"
echo "   cd Backend && python3 run.py"
echo ""
echo "If you still get segmentation faults:"
echo "- Check dmesg for errors: sudo dmesg | tail -50"
echo "- Monitor memory: watch -n 1 free -h"
echo "- Check CUDA: nvidia-smi"
echo ""
