import time
import serial

print "Iniciando Comunicao Serial com Arduino"
# Iniciando conexao serial
comport = serial.Serial('/dev/ttyACM0', 115200)
#comport = serial.Serial('/dev/ttyUSB0', 115200)

LED_ON='l'
LED_OFF='d'

# Time entre a conexao serial e o tempo para escrever (enviar algo)
time.sleep(1.8) # Entre 1.5s a 2s

print "-*- LOOP -*-"
try:
  while True:
    print "Led ON"
    comport.write(LED_ON)
    time.sleep(1)
    print "Led OFF"
    comport.write(LED_OFF)
    time.sleep(1)
except:
  # Fechando conexao serial
  comport.close()
  pass
