#include <ArduinoJson.h>

int vrxPin = A7;  // Pin VRX du joystick
int vryPin = A5;  // Pin VRY du joystick
int swPin = 2;    // Pin SW du joystick
int swButton = 4;    // Pin SW du bouton
int swbutton2 = 7;    // Pin SW du bouton
int gripperPin = A1;

void setup() {
  Serial.begin(9600);   // Initialise la communication série à 9600 bauds
  pinMode(swPin, INPUT_PULLUP);  // Déclare la broche SW comme entrée avec résistance de pull-up
  pinMode(swButton, INPUT_PULLUP);  // Déclare la broche SW comme entrée avec résistance de pull-up
  pinMode(swbutton2, INPUT_PULLUP);  // Déclare la broche SW comme entrée avec résistance de pull-up
}

void loop() {
  int vrxValue = analogRead(vrxPin);  // Lire la valeur de VRX
  int vryValue = analogRead(vryPin);  // Lire la valeur de VRY
  int gripperValue = analogRead(gripperPin);  // Lire la valeur du potentiomètre
  int swValue = digitalRead(swPin);   // Lire l'état du bouton SW
  int swValue2 = digitalRead(swButton);   // Lire l'état du bouton SW
  int swValue3 = digitalRead(swbutton2);   // Lire l'état du bouton SW

  // Envoie les valeurs sous forme de tableau séparé par des virgules
  Serial.print(vrxValue);
  Serial.print(",");
  Serial.print(vryValue);
  Serial.print(",");
  Serial.print(swValue);
  Serial.print(",");
  Serial.print(swValue2);
  Serial.print(",");
  Serial.print(swValue3);
  Serial.print(",");
  Serial.println(gripperValue);

  delay(100); // Un petit délai pour ne pas surcharger la communication
}
