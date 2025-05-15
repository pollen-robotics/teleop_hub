# Teleoperation without VR

A general project combining different methods for teleoperating Reachy2 without virtual reality. 

You can find the article on the development of the project and the technical details at this [link](AJOUT DU Medium link). 

## Methods available

There are currently 5 methods available:
- the controller with the Vive tracker
- the controller with the ArUco cube
- RGBD camera
- the SOARM-100 robotic arm
- the gamepad

They all have their pros and cons, which you can read about in the article, or find out for yourself by testing them. 

## How to install this repository ?

1. Clone this repository 

        git clone blablabla 

2. Install the dependencies, according to which modalities and robot you want to use : 

        pip install -e ".[_modalities_]"
    
    For example, if you want to install the required libraries for Reachy2, Vive Tracker and RGBD Camera : <code> pip install -e ".[reachy2, vive, rgbd]"</code>
    
    The options are : *reachy2*, *vive*, *aruco*, *rgbd*, *so100*, *all*. 

3. Apply the dev rules: 

        sudo cp 11-noVR.setup.rules /etc/udev/rules.d/
        sudo udevadm control --reload-rules && sudo udevadm trigger

> Be careful, if you're using the RGBD modality with the supplied Orbbec class, you need to manually install *pyorbbecsdk*. The instructions are below in the specific [RGBD Camera section](#how-to-set-up-each-modality). 

## How to use it ?
The project configuration file is used to select a particular teleoperation mode. 
You can therefore modify it according to what you want to use. 

To do this, go to the project's config.yaml file:

```
cd noVR_teleoperation
nano config.yaml 
```

The first 5 lines are essential to set-up the project : 

- **robot_ip & fake_only**: 
	
    You can change *localhost* for the ip address of your robot  - *if you don't know how to find your Reachy2's IP address, go [there](https://pollen-robotics.github.io/reachy2-docs/developing-with-reachy-2/getting-started-sdk/connect-reachy2/)*. 

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

> Be careful that the robot will move into the 90° bent position when the teleoperation is launched. So make sure there are **no obstacles** in its path (if it is too close to a table, for example). 

## What about teleoperating other robots ?
For the moment, only Reachy2 is available, as this projet was developed around it, but there are plans to add other robots. 

If you'd like to add your own robot, it is possible ! You have to add a child class to the **Robot** one (in the robot folder), adjusting the various methods for controlling the robot's parts, and change the initialisation of the **Teleoperation** class in the *teleoperation.py* file, at the **self.robot** level. 
Feel free to test it and to suggest your additions!


## How to set-up each modality

<details>
<summary> <span style="font-size: 1.2em;"><strong>Vive Tracker & ArUco cube </strong></span></summary>

These two techniques use the [SOARM-100](https://github.com/TheRobotStudio/SO-ARM100/tree/main) leader joystick.
[à compléter : explications sur l'impression + branchements de la manette]

On the top, you can either screw in a Vive tracker or print out a cube on which you can stick ArUco units. 

<details>
<summary><span style="font-size: 1.1em;"><strong>Vive Tracker </strong></span></summary>

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

    a. <code>gedit ~/.steam/steam/steamapps/common/SteamVR/resources/settings/default.vrsettings</code>
    
    Change the value of "requireHmd" to *false*, "forcedDriver" to *null*, and 'activateMultipleDrivers" to *true*. 

    b. <code>gedit ~/.steam/steam/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings </code>

    Change the value of "enable" to *true*. 
> Be careful that SteamVR updates can sometimes overwrite changes to files. If the base station and trackers are no longer detected, don't hesitate to check that the changes are still there. If not, edit again and restart SteamVR.


</details>


To launch the teleoperation, it needs :
- Running SteamVR
- Vive Base Station plugged, at least 1m away from the trackers, with no obstacles in the way (avoid being too close to a computer, which can interfere with the signal)
- Detected trackers - *You can find out more about pairing trackers on the [Vive website](https://www.vive.com/us/support/tracker3/category_howto/pairing-vive-tracker.html)*

</details>
<details>
<summary><span style="font-size: 1.1em;"><strong>ArUco cube</strong></span></summary>

What you need is :
- an ArUco cube 
- a camera

1. **The cube**

You need to print the provided cube (lien à ajouter), in white for a better detection. Then, you have print the ArUco markers : we use ArUco markers of 6x6 dictionary, from 0 to 10, with a size of 5.5cm (you can use [this site](https://fodi.github.io/arucosheetgen/) to generate the sheet with the markers).  Then you can paste them at the center of each face, as shown in the diagram below, with the cross representing the top left corner of the marker, and in ascending numerical order is back, up, left, down, right, front. 

<p align = "center"> 
    <img src="images/cube_back.png" alt="cube back" style="width: 27.5%; margin-right: 10px;"/>
    <img src="images/cube_front.png" alt="cube front" style="width: 25.5%; margin-right: 10px;"/>
</p>

You can adjust the markers id, numerical order and size directly in the config file, in the ARUCO section. 

2. **The camera**

You can use a camera or your smartphone. To select it, you need to change the config file in the ARUCO section. 

If you're using a camera : 
- set the ubs_mode to *true*
- set the camera_id (the integrated one is '*0*', if it's an external camera, the number is the index in the order of usb-connected devices)

If you're using your smartphone :
- you need to install the app "IP webcam" on your phone, then click on the 3 dots and on "start the server"
- copy the IPv4 address (between http:// and :8080, not included - *it should be something like 172.16.0.56*) and paste it in the camera_id
- make sure your phone and computer are on the same network, and that your phone stays unlocked. 

By default, *with_calibration* parameter is set to *false* in the config file. That means that the intrinsic camera parameters used are estimated manually (which is sufficient for cameras with little distortion such as integrated computer cameras). But you have the possibility to use specific camera matrix and distorsion coefficients, by setting *with_calibration* to true and replacing the *camera_matrix* and *dist_coeffs* with your own values. 
We also provide the script to perform your calibration with a ChArUco board. 

<details>
<summary> Steps to calibrate your camera </summary>

1. Go to the camera_calibration folder :
<code> cd camera_calibration </code>

2. Generate your ChArUco board : 
<code> python3 charuco_generator.py </code>

3. Print it and paste it on a rigid surface

4. Launch the calibration script : <code>python3 camera_calibration.py</code>

By default, the camera taken into account is the one built into the computer, but you can select the one you want to calibrate by adding the --usb_mode (true or false) and -- camera_id (index or IP address) arguments: for example by executing <code> python3 camera_calibration.py --usb-mode false --camera_id "172.16.0.56"</code>. 

Move the board in all directions, the important thing is to have different orientations, close-up shots, distant shots and angled shots. This takes 20 images, in which the board must be placed quite a distance from the previous image. 

Then, it saves the parameters in the camera_parameters folder, but also by updating the teleoperation config file. 

</details>

</details>

#### Both

For the trigger, both of these methods can use either a Feetech-type motor or a potentiometer. 
They both use an Arduino system for the button and joystick data.

It is therefore necessary to enter the type of trigger control and the Arduino +/Feetech ports for your hardware in the config.yaml file. 
[à compléter : explications sur comment on retrouve les noms des ports et changer le fichier config]


</details>

<details>
<summary> <span style="font-size: 1.2em;"><strong> RGBD Camera </strong></span></summary>

The posture detection model is [Mediapipe](https://chuoling.github.io/mediapipe/solutions/holistic.html). 

We use an [Orbbec Femto Bold](https://www.orbbec.com/products/tof-camera/femto-bolt/), but you are free to adapt the code to use your own RGBD Camera. 
To use an Orbbec camera, you need the package pyorbbecsdk.

<details>
<summary><strong>Download pyorbbecsdk : </strong></summary>

1. Clone the repository (virtual environment recommended) : <code> git clone https://github.com/orbbec/pyorbbecsdk.git </code>

2. Make sure you have the needed dependencies : <code> sudo apt-get install python3-dev python3-pip python3-opencv </code>

3. Install the requirements : 
    
    ```
    cd pyorbbecsdk
    pip3 install -r requirements.txt 
    ```

4. Create a folder for the build : 

        mkdir build
        cd build
        cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
        

5. Build the wrapper : 
        
        make -j4
        make install

6. Add the library directory to the list (*to know your python path, write <code> which python </code> in your terminal*): 

    <code>export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/</code>

7. Import and apply dev rules : 

    ```
    sudo bash ./scripts/install_udev_rules.sh
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```

8. Install the library : 
    
    ```
    cd ..
    pip install -e .
    ```

</details>



To set-up your environment : 
- Position the camera high up, to avoid getting occlusion. A calibration will be performed when the script is launched to calculate the orientation and adapt the calculation of the poses. 

- Position yourself with your head straight and your arms bent at 90°.
> Note that the script will wait until your hands are correctly positioned before launching the teleoperation, to avoid any sudden movements by the robot. 

- You can move your head and your two arms, keeping your torso static. You can open and close the grippers by moving your index finger and thumb towards or away from each other. The orientation of the robot's hand is defined by the orientation of the elbow-wrist vector.

- To stop the teleoperation, you have to tilt the head downwards for a prolonged period of time, until the streaming window closes. 


</details>

<details>
<summary> <span style="font-size: 1.2em;"><strong> SOARM-100 </strong></span></summary>


</details>

<details>
<summary> <span style="font-size: 1.2em;"><strong> Gamepad </strong></span></summary>


</details>
