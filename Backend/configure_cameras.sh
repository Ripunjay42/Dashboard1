#!/bin/bash
# Auto-configure USB cameras to MJPEG mode for optimal Jetson performance

echo "🎥 Configuring USB cameras for Jetson Nano/Orin..."
echo ""

# Function to configure a camera
configure_camera() {
    local device=$1
    local cam_num=$2
    
    if [ -e "$device" ]; then
        echo "📹 Camera $cam_num ($device):"
        
        # Check if camera supports MJPEG
        if v4l2-ctl -d "$device" --list-formats-ext | grep -q "MJPG"; then
            echo "   ✅ MJPEG supported"
            
            # Set to MJPEG 640x480 @ 30 FPS
            v4l2-ctl -d "$device" --set-fmt-video=width=640,height=480,pixelformat=MJPG 2>/dev/null
            v4l2-ctl -d "$device" --set-parm=30 2>/dev/null
            
            # Verify configuration
            FORMAT=$(v4l2-ctl -d "$device" --get-fmt-video | grep "Pixel Format" | awk '{print $4}')
            WIDTH=$(v4l2-ctl -d "$device" --get-fmt-video | grep "Width/Height" | awk '{print $3}')
            
            if [ "$FORMAT" = "'MJPG'" ]; then
                echo "   ✅ Set to MJPEG ${WIDTH}/480 @ 30 FPS"
                
                # Test stream stability
                echo "   Testing stream stability..."
                RESULT=$(timeout 3 v4l2-ctl -d "$device" --stream-mmap --stream-count=30 2>&1 | tail -1)
                echo "   $RESULT"
            else
                echo "   ⚠️  Failed to set MJPEG, using default format"
            fi
        else
            echo "   ⚠️  MJPEG not supported, will use YUYV"
        fi
        echo ""
    else
        echo "📹 Camera $cam_num ($device): Not found"
        echo ""
    fi
}

# Configure Camera 0 (Pothole Detection)
configure_camera "/dev/video0" "0"

# Configure Camera 1 (Blind Spot Left - future)
configure_camera "/dev/video1" "1"

# Configure Camera 2 (Blind Spot Right)
configure_camera "/dev/video2" "2"

echo "✅ Camera configuration complete!"
echo ""
echo "📊 Summary:"
v4l2-ctl -d /dev/video0 --get-fmt-video 2>/dev/null | grep -E "(Pixel Format|Width/Height)" | head -2
echo ""
echo "💡 Tip: Run this script after every reboot for optimal performance"
echo "   Add to ~/.bashrc: source ~/Desktop/Experience_centre_Dashboard/Backend/configure_cameras.sh"
