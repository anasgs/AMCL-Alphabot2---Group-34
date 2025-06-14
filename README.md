# AMCL Alphabot2 - Group 34
This project implements the **Adaptive Monte Carlo Localization (AMCL)** algorithm for an **Alphabot2** robot navigating in a known environment using a **grid map**.
It was developed as part of the **Autonomous Systems** course at Instituto Superior Técnico, University of Lisbon, by **Group 34**.

---

## 📁 Project Structure

```bash
.
├── AMCL_algorithm.py           # Core implementation of the AMCL algorithm
├── simulator_sala.py           # Simulator setup: room environment
├── simulador_corredor.py       # Simulator setup: corridor environment
├── alphabot_code/              # Code and scripts to deploy on the AlphaBot2 and PC
│   ├── setup_aruco_detector.sh     # Bash script to set up the camera on the AlphaBot2
│   ├── run_raspicam_320x240.sh     # Bash script to install and run the ArUco detector on PC
│   └── aruco_node.py               # Python ROS node for ArUco detection that is created when run_raspicam_320x240.sh is executed.
├── README.md                   # Project documentation

```
---

## 📦 Packages Required
- Python 3.x
- NumPy
- Matplotlib
- OpenCV (for map handling, ArUco marker detection and image processing)
- Random, math, etc. (standard Python libraries)
- Python 3.x
- ROS 1 Noetic (on both AlphaBot and PC)
- ROS packages (cv_bridge, image_transport, sensor_msgs)

## 👨‍🎓 Authors
- Ana Silva (103013)
- Camila Abreu (102080)
- Catarina Finuras
- Alexandre Frazão (111098)
