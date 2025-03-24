const int speaker_pin = 3;

void setup(){
  pinMode(speaker_pin, OUTPUT);
  Serial.begin(9600);
}
void loop(){
  if(Serial.available() > 0){
    // this command is from python
    char command = Serial.read();
    if(command == '1'){
       tone(speaker_pin, 440, 1000);
    }else if(command == '0'){
        noTone(speaker_pin);
    }
  }
}