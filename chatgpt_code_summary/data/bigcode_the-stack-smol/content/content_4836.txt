import socket
import struct
import os

PACKET_SIZE = 1024
TIME_OUT = 5
SUCCESS = b'File Has Been Transferred'


def getPayload(fileName):
    try:
        with open(file=fileName, mode="r+b") as readFile:
            payload = readFile.read()
            if len(payload) == 0:
                print("That is a blank file.\nProgram now exiting ...")
                exit()

            return payload

    except FileNotFoundError:
        print("\nNo payload file.\nProgram now exiting ...")
        exit()


def main():
    # fileName = "test.txt"
    # serverIP = "127.0.0.1"
    # serverPort = 5005

    fileName = input("Enter path of the file to be sent to the server:\n")
    payload = getPayload(fileName=fileName)
    print("File Found ...")

    serverIP = input("\nEnter the IP Address of the server:\n")
    if serverIP is None:
        print("Cannot leave server IP address blank.\nProgram now exiting ...")
        exit()

    try:
        serverPort = int(input("\nEnter the Port of the server:\n"))
    except ValueError as ve:
        print("Please provide a valid port number. Should only contain character 0-9.\nProgram now exiting ...")
        exit()

    if serverPort is None:
        print("Cannot leave server port blank.\nProgram now exiting ...")
        exit()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(TIME_OUT)

            sock.settimeout(TIME_OUT)
            sock.connect((serverIP, serverPort))

            print("\nTransferring File ...")

            name = fileName.split("/")[-1]
            nameBytes = name.encode("utf-8")
            nameLength = len(nameBytes)

            nameSizeBytes = struct.pack("I", nameLength)
            payloadLength = len(payload) + 8 + nameLength

            numPackets = (payloadLength // PACKET_SIZE)
            if (payloadLength / PACKET_SIZE) > numPackets:
                numPackets += 1

            packedNumBytes = struct.pack('I', numPackets)

            header = packedNumBytes + nameSizeBytes + nameBytes

            payload = header + payload

            sock.sendall(payload)
            data = sock.recv(PACKET_SIZE)

        print("\nStatus:")
        print(data.decode("utf-8"))
        print("\nProgram done ...")

    except ConnectionRefusedError or ConnectionResetError as e:
        print(f"\n{e} Error Occurred. Check for correct server IP Address and Ports. Check server status.\nProgram now exiting ...")

    except Exception as e:
        print(f"\n{e} error has broken things.")


if __name__ == '__main__':
    os.system("clear")
    main()
