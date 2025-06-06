# Assembly guide

This is a DIY controller. Two variants are possible:
- A version using a **Vive Tracker**
- A **low-cost version** using an **ArUco** cube


## Requirements : 

You will need : 
- A 3D printer
- A soldering iron, and basic electronics tools and skills
- Some wire
- some screws :
    - M2.2x5mm x2 
    - M2.5x8mm x4
    - M3x6mm x4
    - M2.2x6mm x2



## BOM



### ArUco Controller (~25 €)

| **Component**            | **Quantity** | **Unit Price (€)** | **Total (€)** | **Link** |
|--------------------------|--------------|---------------------|---------------|----------|
| Arduino Nano Every       | 1            | 2.5                 | 2.5           | [AliExpress](https://fr.aliexpress.com/item/1005007475356474.html) |
| Potentiometer            | 1            | 0.25                | 0.25          | [AliExpress](https://fr.aliexpress.com/item/1005009069826370.html) |
| Push Button              | 2            | 0.1                 | 0.2           | [AliExpress](https://fr.aliexpress.com/item/1005003938244847.html) |
| Joystick Module          | 1            | 1.5                 | 1.5           | [AliExpress](https://fr.aliexpress.com/item/1005005317345964.html) |
| Micro USB Cable          | 1            | 5                   | 5             | [AliExpress](https://fr.aliexpress.com/item/1005006017853917.html) |
| PLA Filament (~130g)     | ~130g        | 4                   | 4             | —        |
| **Total**                |              |                     | **13.45 €**   |          |

> ~30€ for two controllers.
---

### Vive Controller (~266 €)



| **Component**            | **Quantity** | **Unit Price (€)** | **Total (€)** | **Link** |
|--------------------------|--------------|---------------------|---------------|----------|
| Arduino Nano Every       | 1            | 2.5                 | 2.5           | [AliExpress](https://fr.aliexpress.com/item/1005007475356474.html) |
| Push Button              | 2            | 0.1                 | 0.2           | [AliExpress](https://fr.aliexpress.com/item/1005003938244847.html) |
| Joystick Module          | 1            | 1.5                 | 1.5           | [AliExpress](https://fr.aliexpress.com/item/1005005317345964.html) |
| Potentiometer            | 1            | 0.25                | 0.25          | [AliExpress](https://fr.aliexpress.com/item/1005009069826370.html) |
| Micro USB Cable          | 1            | 5                   | 5             | [AliExpress](https://fr.aliexpress.com/item/1005006017853917.html) |
| PLA Filament (~130g)     | ~130g        | 4                   | 4             | —        |
| Vive Tracker             | 1            | 100                 | 100           | [Materiel.net](https://www.materiel.net/produit/202103080068.html?gQT=2) |
| Vive Base Station 2.0    | 1            | 140                 | 140           | [Immersive Display](https://immersive-display.com/fr/htc-vive-pro-eye/553-station-de-base-htc-vive-steamvr-20.html) |
| **Total**                |              |                     | **253.45 €**   |          |


> 💡 One Vive Base Station can track two Vive Trackers → ~372 € for two controllers.



## 3D Printed Parts

You can find all the STL files in the [`project_resources` folder](../STL_files/potentiometer_controller/)

For **each controller**, print the following:

### Core parts:
- `side`_potentiometer_controller ×1  
- `side`_potentiometer_handle1 ×1  
- `side`_potentiometer_handle2 ×1  
- `side`_trigger ×1
- button ×2  
- button_support ×1  

### Tracking-specific top:
- **ArUco version:** `side`_aruco_controller
- **Vive version:** `side`_vive_controller

> Replace  `side` by `left`/`right` based on the controller side you want.


## Assembly Instructions

### 1. Print the controller parts

Use your preferred slicer and printer to produce the components listed above.

### 2. Assemble the controller

- Insert the push-buttons into their holders
  <p align="center">
    <img src="../../images/assembly/buttons1.png" width="45%">
    <img src="../../images/assembly/buttons2.png" width="45%">
  </p>
> 💡 you can add hot glue to secure them in place

- Add the two buttons on top of the push-button
<p align="center">
    <img src="../../images/assembly/buttons3.png" width="50%">
</p>

- Screw the button-holder to the handle with two PLATCB M2.2 × 5 plastic-tapping screws.
<p align="center">
    <img src="../../images/assembly/buttons4.png" width="50%">
</p>

- Position the joystick and fasten it using one PLATCB M3 × 6 plastic-tapping screw.

<p align="center">
    <img src="../../images/assembly/joystick1.png" width="50%">
</p>

- Insert the potentiometer into its slot and secure it using the nut provided.  
  <p align="center">
    <img src="../../images/assembly/potentiometer1.png" width="50%">
  </p>

- Complete the wiring exactly as shown in the schematic.
<p align="center">
    <img src="../../images/assembly/schema_potentiometer.png" width="50%">
</p>

- Fix the Arduino board to its standoffs

<p align="center">
    <img src="../../images/assembly/arduino1.png" width="50%">
</p>

- Close the handle, tightening the two halves together with four M2.5x8 x4 screws.
<p align="center">
    <img src="../../images/assembly/assembly1.png" width="50%">
</p>


- Fix the potentiometer cap to the trigger.  
  <p align="center">
    <img src="../../images/assembly/trigger1.png" width="50%">
  </p>

- Attach the cap (with the trigger) onto the potentiometer shaft.  
  <p align="center">
    <img src="../../images/assembly/trigger2.png" width="50%">
  </p>

- Slide the upper part of the controller into the handle until fully inserted. Secure both parts with a screw.  
  <p align="center">
    <img src="../../images/assembly/potentiometer2.png" width="50%">
  </p>

- **For vive only** : Fix the Vive Tracker to the top of the controller.  
  If you don’t have the appropriate screw, you can use the 3D-printed mount provided.  
  <p align="center">
    <img src="../../images/assembly/vive_potentiometer.png" width="50%">
  </p>


#### Final assembly

- Slide the tracker part onto the controller — Once fully inserted, secure them together by adding M2.2x6mm screw at the designated point.
<p align="center"> 
  <img src="../../images/assembly/potentiometer_assembly1.png" width="45%"> 
  <img src="../../images/assembly/potentiometer_assembly2.png" width="45%"> 
</p>



### 3. Upload the code

- Open the [PlatformIO project](../arduino_code/potentiometer/)
- Connect the Arduino Nano to your computer via USB
- Upload the firmware to the board

---