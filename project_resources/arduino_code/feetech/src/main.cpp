#include <ArduinoJson.h>

int vrxPin = A7;
int vryPin = A5;
int swPin = 2;
int swButton = 4;
int swbutton2 = 7;

void setup()
{
  Serial.begin(9600);               
  pinMode(swPin, INPUT_PULLUP);    
  pinMode(swButton, INPUT_PULLUP);  
  pinMode(swbutton2, INPUT_PULLUP); 
}

void loop()
{
  int vrxValue = analogRead(vrxPin);
  int vryValue = analogRead(vryPin);
  int swValue = digitalRead(swPin);
  int swValue2 = digitalRead(swButton);
  int swValue3 = digitalRead(swbutton2);

  Serial.print(vrxValue);
  Serial.print(",");
  Serial.print(vryValue);
  Serial.print(",");
  Serial.print(swValue);
  Serial.print(",");
  Serial.print(swValue2);
  Serial.print(",");
  Serial.println(swValue3);
  delay(100);
}
