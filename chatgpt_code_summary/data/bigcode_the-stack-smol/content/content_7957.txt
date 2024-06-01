"""This uses the CLUE as a Bluetooth LE sensor node."""
 
# Adafruit Service demo for Adafruit CLUE board.
# Accessible via Adafruit Bluefruit Playground app and Web Bluetooth Dashboard.

import time

import board
from digitalio import DigitalInOut
import neopixel_write
from adafruit_ble import BLERadio

import ulab

from adafruit_clue import clue

from adafruit_ble_adafruit.adafruit_service import AdafruitServerAdvertisement

from adafruit_ble_adafruit.accelerometer_service import AccelerometerService
from adafruit_ble_adafruit.addressable_pixel_service import AddressablePixelService
from adafruit_ble_adafruit.barometric_pressure_service import BarometricPressureService
from adafruit_ble_adafruit.button_service import ButtonService
from adafruit_ble_adafruit.humidity_service import HumidityService
from adafruit_ble_adafruit.light_sensor_service import LightSensorService
from adafruit_ble_adafruit.microphone_service import MicrophoneService
from adafruit_ble_adafruit.temperature_service import TemperatureService
from adafruit_ble_adafruit.tone_service import ToneService

accel_svc = AccelerometerService()
accel_svc.measurement_period = 100
accel_last_update = 0

# CLUE has just one board pixel. 3 RGB bytes * 1 pixel.
NEOPIXEL_BUF_LENGTH = 3 * 1
neopixel_svc = AddressablePixelService()
neopixel_buf = bytearray(NEOPIXEL_BUF_LENGTH)
# Take over NeoPixel control from clue.
clue._pixel.deinit()  # pylint: disable=protected-access
neopixel_out = DigitalInOut(board.NEOPIXEL)
neopixel_out.switch_to_output()

baro_svc = BarometricPressureService()
baro_svc.measurement_period = 100
baro_last_update = 0

button_svc = ButtonService()
button_svc.set_pressed(False, clue.button_a, clue.button_b)

humidity_svc = HumidityService()
humidity_svc.measurement_period = 100
humidity_last_update = 0

light_svc = LightSensorService()
light_svc.measurement_period = 100
light_last_update = 0

# Send 256 16-bit samples at a time.
MIC_NUM_SAMPLES = 256
mic_svc = MicrophoneService()
mic_svc.number_of_channels = 1
mic_svc.measurement_period = 100
mic_last_update = 0
mic_samples = ulab.zeros(MIC_NUM_SAMPLES, dtype=ulab.uint16)

temp_svc = TemperatureService()
temp_svc.measurement_period = 100
temp_last_update = 0

tone_svc = ToneService()

clue_display = clue.simple_text_display(text_scale=3, colors=(clue.WHITE,))
 
clue_display[0].text = "Temperature &"
clue_display[1].text = "Humidity"
clue_display[3].text = "Temp: {:.1f} C".format(clue.temperature)
clue_display[5].text = "Humi: {:.1f} %".format(clue.humidity)

ble = BLERadio()
# The Web Bluetooth dashboard identifies known boards by their
# advertised name, not by advertising manufacturer data.
ble.name = "Attic"

# The Bluefruit Playground app looks in the manufacturer data
# in the advertisement. That data uses the USB PID as a unique ID.
# Adafruit CLUE USB PID:
# Arduino: 0x8071,  CircuitPython: 0x8072, app supports either
adv = AdafruitServerAdvertisement()
adv.pid = 0x8072

while True:
    # Advertise when not connected.
    ble.start_advertising(adv)
    while not ble.connected:
        pass
    ble.stop_advertising()

    while ble.connected:
        now_msecs = time.monotonic_ns() // 1000000  # pylint: disable=no-member

        if now_msecs - accel_last_update >= accel_svc.measurement_period:
            accel_svc.acceleration = clue.acceleration
            accel_last_update = now_msecs

        if now_msecs - baro_last_update >= baro_svc.measurement_period:
            baro_svc.pressure = clue.pressure
            baro_last_update = now_msecs

        button_svc.set_pressed(False, clue.button_a, clue.button_b)

        if now_msecs - humidity_last_update >= humidity_svc.measurement_period:
            humidity_svc.humidity = clue.humidity
            humidity_last_update = now_msecs
            clue_display[5].text = "Humi: {:.1f} %".format(clue.humidity)
            print("Humi: {:.1f} %".format(clue.humidity))


        if now_msecs - light_last_update >= light_svc.measurement_period:
            # Return "clear" color value from color sensor.
            light_svc.light_level = clue.color[3]
            light_last_update = now_msecs

        if now_msecs - mic_last_update >= mic_svc.measurement_period:
            clue._mic.record(  # pylint: disable=protected-access
                mic_samples, len(mic_samples)
            )
            # This subtraction yields unsigned values which are
            # reinterpreted as signed after passing.
            mic_svc.sound_samples = mic_samples - 32768
            mic_last_update = now_msecs

        neopixel_values = neopixel_svc.values
        if neopixel_values is not None:
            start = neopixel_values.start
            if start > NEOPIXEL_BUF_LENGTH:
                continue
            data = neopixel_values.data
            data_len = min(len(data), NEOPIXEL_BUF_LENGTH - start)
            neopixel_buf[start : start + data_len] = data[:data_len]
            if neopixel_values.write_now:
                neopixel_write.neopixel_write(neopixel_out, neopixel_buf)

        if now_msecs - temp_last_update >= temp_svc.measurement_period:
            temp_svc.temperature = clue.temperature
            temp_last_update = now_msecs
            clue_display[3].text = "Temp: {:.1f} C".format(clue.temperature)
            print("Temp: {:.1f} C".format(clue.temperature))


        tone = tone_svc.tone
        if tone is not None:
            freq, duration_msecs = tone
            if freq != 0:
                if duration_msecs != 0:
                    # Note that this blocks. Alternatively we could
                    # use now_msecs to time a tone in a non-blocking
                    # way, but then the other updates might make the
                    # tone interval less consistent.
                    clue.play_tone(freq, duration_msecs / 1000)
                else:
                    clue.stop_tone()
                    clue.start_tone(freq)
            else:
                clue.stop_tone()
        last_tone = tone
        clue_display.show()
        time.sleep(5)


# import time
# from adafruit_clue import clue
# import adafruit_ble_broadcastnet
 
# print("This is BroadcastNet CLUE sensor:", adafruit_ble_broadcastnet.device_address)
 
# while True:
#     measurement = adafruit_ble_broadcastnet.AdafruitSensorMeasurement()
 
#     measurement.temperature = clue.temperature
#     measurement.pressure = clue.pressure
#     measurement.relative_humidity = clue.humidity
#     measurement.acceleration = clue.acceleration
#     measurement.magnetic = clue.magnetic
 
#     print(measurement)
#     adafruit_ble_broadcastnet.broadcast(measurement)
#     time.sleep(5)

# """This uses the CLUE as a Bluetooth LE sensor node."""

# import time
# from adafruit_clue import clue
# from adafruit_ble import BLERadio
# from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
# from adafruit_ble.services.nordic import UARTService

# ble = BLERadio()
# ble.name = "patio"
# uart_server = UARTService()
# advertisement = ProvideServicesAdvertisement(uart_server)


# while True:
# #     measurement = adafruit_ble.advertising.AdafruitSensorMeasurement()

# #     measurement.temperature = clue.temperature
# #     measurement.pressure = clue.pressure
# #     measurement.relative_humidity = clue.humidity
# #     measurement.acceleration = clue.acceleration
# #     measurement.magnetic = clue.magnetic
#     print("{},{},{}\n".format(clue.temperature-5,clue.humidity,clue.pressure))

#     # Advertise when not connected.
#     ble.start_advertising(advertisement)
#     print(advertisement)
#     while not ble.connected:
#         pass
#     ble.stop_advertising()

#     while ble.connected:
#         print("{},{},{}\n".format(clue.temperature-5,clue.humidity,clue.pressure))
#         uart_server.write("{},{},{}\n".format(clue.temperature-5,clue.humidity,clue.pressure))
#         time.sleep(15)

#     #time.sleep(1)