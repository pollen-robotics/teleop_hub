import serial
import time

# Ouvre la connexion série avec l'Arduino (change '/dev/ttyUSB0' si nécessaire)
ser = serial.Serial('/dev/noVR_left_arduino', 9600)  # Remplace par ton port série

ser2 = serial.Serial('/dev/noVR_right_arduino', 9600)
time.sleep(2)  # Attends que la connexion soit établie

while True:
    if ser.in_waiting > 0:  # Vérifie si des données sont disponibles
        data = ser.readline().decode('utf-8').strip()  # Lire la ligne et la décoder
        values = data.split(',')  # Découpe la ligne en une liste de valeurs

        # Convertir chaque valeur en entier
        vrx = int(values[0])
        vry = int(values[1])
        sw = int(values[2])
        sw2 = int(values[3])
        sw3 = int(values[4])

        # Afficher les valeurs sous forme de tableau
        print(f" ______________ VRX: {vrx}, VRY: {vry}, SW: {sw}", f"SW2: {sw2}, SW3: {sw3}")

    if ser2.in_waiting > 0:
        data = ser2.readline().decode('utf-8').strip()  # Lire la ligne et la décoder
        values = data.split(',')  # Découpe la ligne en une liste de valeurs

        # Convertir chaque valeur en entier
        vrx = int(values[0])
        vry = int(values[1])
        sw = int(values[2])
        sw2 = int(values[3])
        sw3 = int(values[4])

        # Afficher les valeurs sous forme de tableau
        print(f" @@@@@@@@@@@@ VRX: {vrx}, VRY: {vry}, SW: {sw}", f"SW2: {sw2}, SW3: {sw3}")