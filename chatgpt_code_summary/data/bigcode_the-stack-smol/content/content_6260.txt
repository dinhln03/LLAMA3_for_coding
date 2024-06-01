import bthomehub

client = bthomehub.BtHomeClient('192.168.0.254')

client.authenticate()

values = client.get_values()

print('DownstreamRate = ' + str(float(values["Device/DSL/Channels/Channel[@uid='1']/DownstreamCurrRate"]) / 1000))
print('UpstreamRate = ' + str(float(values["Device/DSL/Channels/Channel[@uid='1']/UpstreamCurrRate"]) / 1000))
print('System UpTime = ' + str(values["Device/DeviceInfo/UpTime"]))
print('BytesSent = ' + str(float(values["Device/IP/Interfaces/Interface[@uid='3']/Stats/BytesSent"]) / 1000000))
print('BytesRecieved = ' + str(float(values["Device/IP/Interfaces/Interface[@uid='3']/Stats/BytesReceived"]) / 1000000))
print('Network UpTime = ' + str(values["Device/IP/Interfaces/Interface[@uid='3']/LastChange"]))

