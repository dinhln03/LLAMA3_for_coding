# Cliente
import socket
HOST = socket.gethostname() # Endereco IP do Servidor
PORT = 9999 # Porta que o Servidor esta
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dest = (HOST, PORT)
print('Para sair, digite $ + Enter')
msg = input("Entre com a mensagem:\n")
udp.sendto (msg.encode('utf-8'), dest)
while msg != '$': # terminação do programa
    (msg, servidor) = udp.recvfrom(1024)
    print(servidor, msg.decode('utf-8'))
    msg = input()
    udp.sendto (msg.encode('utf-8'), dest)
udp.close()