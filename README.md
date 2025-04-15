# reachy2_noVR_teleoperation
```bash
sudo cp 11-noVR.setup.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```


# Docker

docker build -t teleop_tracker -f Docker/Dockerfile .

## ArUco cube
Working in progress. 

As a reminder, this is how the cube needs to be set : 


![Cube_faces](/images/cube_faces.png)

    aruco 0 to 5 : right arm (white cube)
    aruco 6 to 10 : left arm (black cube)
    faces 5 and 10 front 

