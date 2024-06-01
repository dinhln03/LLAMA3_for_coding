import pyowm

owm = pyowm.OWM('ce688b67bbf90c2a0236d4eb23d8c7bd')  # You MUST provide a valid API key


# Will it be sunny tomorrow at this time in Milan (Italy) ?
#forecast = owm.daily_forecast('panama')
tomorrow = pyowm.timeutils.tomorrow()
#forecast.will_be_sunny_at(tomorrow)  # Always True in Italy, right? ;-)

# Search for current weather in London (UK)
observation = owm.weather_at_place('ayangue')
w = observation.get_weather()

rete=w.get_reference_time(timeformat='date')
refe=w.get_reference_time('iso')
estatus=w.get_status()
time=w.get_sunset_time('iso')
wind=(w.get_wind()['speed'])
wind1=w.get_wind()
tempe=w.get_temperature('celsius')
tempe1=w.get_temperature('celsius')['temp_max']
l = observation.get_location()
lugar = l.get_country()

                              # status=Clouds>

# Weather details
#print(forecast)
print(lugar)
print(w)  # <Weather - reference time=2013-12-18 09:20,
#print(rete)
#print(time)
print(estatus)# status=Clouds>
print(refe)
#print(tomorrow)#Dia de mañana
print(wind)#velocidad de viento
print(wind1)
print(w.get_humidity())#humedad
print(tempe)#temperatura
print(tempe1)
#w.get_wind()                  # {'speed': 4.6, 'deg': 330}
#w.get_humidity()              # 87
#w.get_temperature('celsius')  # {'temp_max': 10.5, 'temp': 9.7, 'temp_min': 9.0}