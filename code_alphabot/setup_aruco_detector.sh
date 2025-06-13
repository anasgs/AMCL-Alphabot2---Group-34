#!/bin/bash

echo "📷 Starting or installing raspicam_node with 320x240 resolution..."

# === Path to check if already set up ===
LAUNCH_FILE=~/catkin_ws/src/raspicam_node/launch/camerav2_320x240.launch

# === Already set up? Just run ===
if [ -f "$LAUNCH_FILE" ]; then
	echo "✅ Launch file already exists. Running camera..."
	source ~/catkin_ws/devel/setup.bash
	roslaunch raspicam_node camerav2_320x240.launch
	exit 0
fi

echo "🚧 First-time setup detected. Installing requirements..."

# === 1. Install ROS camera dependencies ===
sudo apt update && sudo apt install -y \
	ros-noetic-camera-info-manager \
	ros-noetic-image-transport \
	ros-noetic-compressed-image-transport \
	ros-noetic-cv-bridge \
	libraspberrypi-dev \
	build-essential \
	cmake \
	git

# === 2. Create catkin workspace ===
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make

# === 3. Clone raspicam_node ===
cd ~/catkin_ws/src
if [ ! -d raspicam_node ]; then
	git clone https://github.com/UbiquityRobotics/raspicam_node.git
fi

# === 4. Create 320x240 launch file ===
mkdir -p ~/catkin_ws/src/raspicam_node/launch
cat > "$LAUNCH_FILE" << 'EOF'
<launch>
  <node name="raspicam_node" pkg="raspicam_node" type="raspicam_node" output="screen">
	<param name="width" value="320" />
	<param name="height" value="240" />
	<param name="fps" value="30" />
	<param name="frame_id" value="camera" />
	<param name="camera_name" value="raspicam" />
	<param name="enable_raw" value="true" />
	<param name="enable_preview" value="false" />
  </node>
</launch>
EOF

# === 5. Build workspace ===
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# === 6. Run the camera ===
echo "✅ Setup complete. Launching 320x240 camera stream..."
roslaunch raspicam_node camerav2_320x240.launch
