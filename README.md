# reachy2_noVR_teleoperation

# Teleoperation without VR

A general project combining different methods for teleoperating Reachy2 without virtual reality. 

You can find the article on the development of the project and the technical details at this [link](Medium link). 

## Methods available

There are currently 5 methods available:
- the console controller
- RGBD camera
- the SOARM-100 robotic arm
- the controller with the Vive tracker
- the controller with the ArUco cube

They all have their pros and cons, which you can read about in the article, or find out for yourself by testing them. 

## How to download this repository ?

... 

## How to use it ?
The project configuration file is used to select a particular teleoperation mode. 
You can therefore modify it according to what you want to use. 

To do this, go to the project's config.yaml file:

```
cd project_tracker
nano config.yaml 
```

The first 5 lines are essential to set-up the project : 

- **robot_ip & fake_only**: 
	
    You can change *localhost* for the ip address of your robot  - *if you don't know how to find your Reachy2's IP adgress, go [there](https://pollen-robotics.github.io/reachy2-docs/developing-with-reachy-2/getting-started-sdk/connect-reachy2/)*. 

	The fake_only parameter is a security that ensures that you only connect to a robot in simulation mode. 
    > If you enter an IP address that corresponds to a robot in real mode (i.e. it is the physical robot that is supposed to be moving), the connection will not be made. 
	If you want to connect to a robot in normal mode, you need to disable this security by changing the fake_only parameter to 'false'.

- **tracker_type**: the modality you want to use (among "aruco", "vive", "rgbd", "arm", "controller")

- **control_mode**: some of the methods can be used to control either a single arm or both arms (Tracker Vive, ArUco cube, SOARM-100): you therefore need to specify *‘dual_arm’* or *‘l_arm’* (for the left arm) or *‘r_arm’* (for the right arm).

- **mirror_mode** : if mirror_mode is set to *true*, you can control the robot in mirror mode, i.e. you can control it face to face, your right arm will control its left arm and vice versa. Otherwise, the robot will be controlled normally.

The rest of the parameters are specific to each teleoperation technique, and we will describe the set-up of each one in details below. 
> *Don't forget to CTRL+X then Y to save the config.yaml and exit*

Then you can launch the teleoperation program :
<code>python -m teleoperation </code>

## What about teleoperating other robots ?
For the moment, only Reachy2 is available, as this projet was developed around it, but there are plans to add other robots. 

If you'd like to add your own robot, it is possible ! You have to add a child class to the **Robot** one (in the robot folder), adjusting the various methods for controlling the robot's parts, and change the initialisation of the **Teleoperation** class in the *teleoperation.py* file, at the **self.robot** level. 
Feel free to test it and to suggest your additions!


## How to set-up each modality ?

### Vive Tracker & ArUco cube 

These two techniques use the [SOARM-100](https://github.com/TheRobotStudio/SO-ARM100/tree/main) leader joystick.
[à compléter : explications sur l'impression + branchements de la manette]

On the top, you can either screw in a Vive tracker or print out a cube on which you can stick ArUco units. 

#### Vive Tracker 

This modality requires 1 Vive tracker for each controller used (it is possible to teleoperate one or both arms), a Vive base station for it to be detected and SteamVR to get data. 

<details>
<summary> Download <b>Steam</b> & <b>SteamVR</b> and enable the headset-free mode:</summary>

1. Install Steam, as well as Python and OpenVR dependencies (we recommend to do it in a virtual environment)

<code> sudo apt-get install steam libsdl2-dev libvulkan-dev libudev-dev libssl-dev zlib1g-dev python-pip</code>


2. Make Steam account & Log in.

3. Install SteamVR : click on Library > VR > Tools > SteamVR

4. Make a Symbolic Link from libudev.so.0 to libudev.so.1 for SteamVR to use

<code>sudo ln -s /lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libudev.so.0</code>

5. Install pyopenvr :  <code>sudo pip install -U pip openvr</code>

6. Disable the headset requirement : 
there are 2 files to modify using those commands on a terminal : 
> <code>gedit ~/.steam/steam/steamapps/common/SteamVR/resources/settings/default.vrsettings</code>
>
> Change the value of "requireHmd" to *false*, "forcedDriver" to *null*, and 'activateMultipleDrivers" to *true*. 
> 
> <code>gedit ~/.steam/steam/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings </code>
>
> Change the value of "enable" to *true*. 

</details>


To launch the teleoperation, it needs :
- Running SteamVR
- Vive Base Station plugged, at least 1m away from the trackers, with no obstacles in the way (avoid being too close to a computer, which can interfere with the signal)
- Detected trackers - *You can find out more about pairing trackers on the [Vive website](https://www.vive.com/us/support/tracker3/category_howto/pairing-vive-tracker.html)*

#### ArUco cube

What you need is :
- an ArUco cube 
- a camera

1. The cube

You need to print the provided cube (lien à ajouter), in white for a better detection. Then, you have print the ArUco markers : we use ArUco markers of 6x6 dictionary, from 0 to 10, with a size of 5.5cm (you can use [this site](https://fodi.github.io/arucosheetgen/) to generate the sheet with the markers).  Then you can paste them at the center of each face, as shown in the diagram below, with the cross representing the top left corner of the marker, and in ascending numerical order is back, up, left, down, right, front. 

You can adjust the markers id, numerical order and size directly in the config file, in the ARUCO section. 

2. The camera

You can use a camera or your smartphone. To select it, you need to change the config file in the ARUCO section. 

If you're using a camera : 
- set the ubs_mode to true
- set the camera_id (the integrated one is '0', if it's an external camera, the number is the index in the order of usb-connected devices)

If you're using your smartphone :
- you need to install the app "IP webcam" on your phone, then click on the 3 dots and on "start the server"
- copy the IPv4 address (between http:// and :8080, not included - *it should be something like 172.16.0.56*) and paste it in the camera_index
- make sure your phone and computer are on the same network, and that your phone stays unlocked. 

You have the possibility to calibrate your camera with a ChArUco Board (configurable [here](https://calib.io/pages/camera-calibration-pattern-generator)) to fine-tune your camera's intrinsic parameters, such as intrinsic matrix and distortion coefficients. 

[AJOUTER LA METHODE]

This automatically saves the parameters in the teleoperation configuration file.

To use these parameters, the with_calibration boolean in this same file must be set to true. Otherwise, manual parameters are applied. 



#### Both

For the trigger, both of these methods can use either a Feetech-type motor or a potentiometer. 
They both use an Arduino system for the button and joystick data.

It is therefore necessary to enter the type of trigger control and the Arduino +/Feetech ports for your hardware in the config.yaml file. 
[à compléter : explications sur comment on retrouve les noms des ports et changer le fichier config]






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

