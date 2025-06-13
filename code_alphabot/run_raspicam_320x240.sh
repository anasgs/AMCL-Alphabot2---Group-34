#!/bin/bash

echo "🔧 Setting up ArUco Detector on a fresh AlphaBot..."

# === 1. Install Required Packages ===
echo "📦 Installing system and ROS dependencies..."
sudo apt update && sudo apt install -y \
  ros-noetic-ros-base \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-camera-info-manager \
  ros-noetic-vision-opencv \
  ros-noetic-rqt-image-view \
  ros-noetic-compressed-image-transport \
  ros-noetic-tf \
  python3-opencv \
  python3-numpy \
  python3-pip \
  python3-yaml \
  git \
  build-essential \
  cmake \
  libraspberrypi-dev

# === 2. Create and Build ROS Workspace ===
echo "🧱 Creating ROS workspace..."
mkdir -p ~/ros/catkin_ws/src
cd ~/ros/catkin_ws
catkin_make

# === 3. Source ROS workspace ===
if ! grep -q "source ~/ros/catkin_ws/devel/setup.bash" ~/.bashrc; then
  echo "source ~/ros/catkin_ws/devel/setup.bash" >> ~/.bashrc
fi
source ~/ros/catkin_ws/devel/setup.bash

# === 4. Clone raspicam_node if not present ===
cd ~/ros/catkin_ws/src
if [ ! -d raspicam_node ]; then
  echo "📥 Cloning raspicam_node..."
  git clone https://github.com/UbiquityRobotics/raspicam_node.git
fi

# === 5. Create 320x240 launch file for raspicam ===
mkdir -p ~/ros/catkin_ws/src/raspicam_node/launch
cat > ~/ros/catkin_ws/src/raspicam_node/launch/camerav2_320x240.launch <<EOF
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

# === 6. Create aruco_detector package ===
cd ~/ros/catkin_ws/src
if [ ! -d aruco_detector ]; then
  echo "📦 Creating aruco_detector package..."
  catkin_create_pkg aruco_detector rospy std_msgs sensor_msgs geometry_msgs cv_bridge image_transport tf
fi

# === 7. Write aruco_node.py script ===
mkdir -p ~/ros/catkin_ws/src/aruco_detector/scripts
cat > ~/ros/catkin_ws/src/aruco_detector/scripts/aruco_node.py << 'EOF'
#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import yaml
import subprocess
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from geometry_msgs.msg import Pose, PoseArray

class ArucoDetectorNode:
	def __init__(self):
    	rospy.loginfo("🟢 Initializing ArUco detector node...")
    	self.bridge = CvBridge()
    	self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    	self.parameters = cv2.aruco.DetectorParameters_create()
    	self.marker_length = 0.05

    	calib_root = os.path.expanduser("~/ros/raspicam_calibrations")
    	os.makedirs(calib_root, exist_ok=True)

    	folders = sorted([d for d in os.listdir(calib_root) if d.startswith("calib_")])
    	if not folders:
        	print("❌ No calibration folders found.")
        	exit(1)

    	selected = folders[0]
    	ost_path = os.path.join(calib_root, selected, "ost.yaml")
    	print(f"📥 Loading calibration from: {ost_path}")
    	with open(ost_path, 'r') as f:
        	calib = yaml.safe_load(f)

    	K = np.array(calib['camera_matrix']['data']).reshape(3, 3)
    	D = np.array(calib['distortion_coefficients']['data'])
    	self.camera_matrix = K.astype(np.float32)
    	self.dist_coeffs = D.astype(np.float32)

    	self.sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback)
    	self.pub_annotated = rospy.Publisher("/aruco/image_annotated", Image, queue_size=1)
    	self.pub_ids = rospy.Publisher("/aruco/marker_ids", Int32MultiArray, queue_size=1)
    	self.pub_distances = rospy.Publisher("/aruco/marker_distances", Float32MultiArray, queue_size=1)
    	self.pub_poses = rospy.Publisher("/aruco/marker_poses", PoseArray, queue_size=1)

    	rospy.loginfo("✅ ArUco detector node ready.")

	def image_callback(self, msg):
    	try:
        	np_arr = np.frombuffer(msg.data, np.uint8)
        	cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    	except Exception as e:
        	rospy.logerr(f"❌ Image decode error: {e}")
        	return

    	corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.dictionary, parameters=self.parameters)
    	pose_array = PoseArray()
    	pose_array.header.stamp = rospy.Time.now()
    	pose_array.header.frame_id = "camera"

    	if ids is not None and len(ids) > 0:
        	cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
        	self.pub_ids.publish(Int32MultiArray(data=ids.flatten().tolist()))

        	rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            	corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

        	distances = []
        	for i, marker_id in enumerate(ids.flatten()):
            	tvec = tvecs[i][0]
            	rvec = rvecs[i][0]
            	distance = np.linalg.norm(tvec)
            	distances.append(distance)
            	rospy.loginfo(f"🧭 Marker {marker_id} → Distance = {distance:.2f} m")

            	R, _ = cv2.Rodrigues(rvec)
            	T = np.eye(4)
            	T[:3, :3] = R
            	quat = tf.transformations.quaternion_from_matrix(T)

            	pose = Pose()
            	pose.position.x = tvec[0]
            	pose.position.y = tvec[1]
            	pose.position.z = tvec[2]
            	pose.orientation.x = quat[0]
            	pose.orientation.y = quat[1]
            	pose.orientation.z = quat[2]
            	pose.orientation.w = quat[3]
            	pose_array.poses.append(pose)

            	cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length)

        	self.pub_distances.publish(Float32MultiArray(data=distances))
        	self.pub_poses.publish(pose_array)
    	else:
        	rospy.loginfo("🔍 No markers detected.")
        	self.pub_ids.publish(Int32MultiArray(data=[]))
        	self.pub_distances.publish(Float32MultiArray(data=[]))
        	self.pub_poses.publish(PoseArray(header=pose_array.header, poses=[]))

    	self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))

if __name__ == "__main__":
	rospy.init_node("aruco_detector_node")
	ArucoDetectorNode()
	rospy.spin()
EOF

chmod +x ~/ros/catkin_ws/src/aruco_detector/scripts/aruco_node.py

# === 8. Build everything ===
cd ~/ros/catkin_ws
catkin_make
source devel/setup.bash

mkdir -p ~/ros/raspicam_calibrations

echo ""
echo "✅ Setup complete!"
echo "▶️ Launch camera: roslaunch raspicam_node camerav2_320x240.launch"
echo "▶️ Then run: rosrun aruco_detector aruco_node.py"
echo ""
