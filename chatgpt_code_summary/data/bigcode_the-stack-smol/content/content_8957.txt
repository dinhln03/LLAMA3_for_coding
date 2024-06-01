from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 53000

client = SimpleUDPClient(ip, port)

client.send_message("/new", "text")
client.send_message("/cue/selected/text/format/color", [1, 1, 0, 1])

