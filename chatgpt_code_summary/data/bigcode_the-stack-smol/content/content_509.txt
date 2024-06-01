import socket
import sys
from config import ip, port

net = 0
sock = None
try:
    if sys.argv[1] == '--connect':
        sock = socket.socket()
        try:
            sock.connect((sys.argv[2], int(sys.argv[3])))
            print('Подключение к игре установлено.')
        except:
            print(f'Неудалось подключиться к игре по адресу {sys.argv[2]}:{sys.argv[3]}')
        net = 1
    elif sys.argv[1] == '--server':
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind((ip, port))
            server.listen(1)
            print(f'Сервер запущен по адресу {ip}:{port}...')
            sock, address = server.accept()
            print(f'Клиент {address[0]}:{address[1]} подключился')
        except:
            print(f'Неудалось запустить сервер по адресу {ip}:{port}')
        net = 2
    else:
        print(f'Неизвестный параметр \'{sys.argv[1]}\'')
except:
    print('Запускается одиночная игра на одном экране')
