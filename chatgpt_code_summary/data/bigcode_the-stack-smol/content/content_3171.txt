import struct
import socket
import ipaddress
from .utils import calculate_checksum

IPV4_HEAD_FMT="!BBHHHBBHII" #H is unsigned short (2 bytes) ! is for network (big-endian)

class IPV4Datagram:
    """
    This class contains 20 bytes IPV4 Datagram
    https://en.wikipedia.org/wiki/IPv4
    |0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|
    ---------------------------------------------------------------------------------------
    |version|  IHL  |      DSCP     | ECN |                Total Length                   |
    ---------------------------------------------------------------------------------------
    |    identification                   | flags  |           Fragemnt Offset            |
    ---------------------------------------------------------------------------------------
    |       TTL     |     Protocol        |               Header Checksum                 |
    ---------------------------------------------------------------------------------------
    |                                Source Ip Address                                    |
    ---------------------------------------------------------------------------------------
    |                                Destination Ip Address                               |
    ---------------------------------------------------------------------------------------
    """

    def __init__(self, source_ip="1.1.1.1",destination_ip="1.1.1.1" , version=4, ihl=5, tos=0,identification=54321,fragment_offset = 0,
                 ttl=253,protocol = socket.IPPROTO_UDP,data='', checksum=0):
        self.version = version
        self.ihl = ihl
        self.version_ihl =  (self.version << 4)  + self.ihl
        self.tos = tos
        self.identification=identification
        self.fragment_offset = fragment_offset
        self.ttl = ttl
        self.protocol = protocol
        self.checksum = checksum
        self.source_ip =int(ipaddress.IPv4Address( source_ip )) # convert into integer
        self.destination_ip = int(ipaddress.IPv4Address(destination_ip ))
        self.data = data
        self.length= 4 * self.ihl + len(self.data)

    def __repr__(self):
        return 'ICMPDatagram({},{},({},{}))'.format(self.type,self.code,self.checksum, self.data)

    def pack(self):
        ipv4_header = struct.pack(IPV4_HEAD_FMT, self.version_ihl,self.tos,self.length, self.identification,
        self.fragment_offset, self.ttl, self.protocol, self.checksum, self.source_ip, self.destination_ip)
        self.checksum = calculate_checksum(ipv4_header)
        ipv4_header = struct.pack(IPV4_HEAD_FMT, self.version_ihl,self.tos,self.length, self.identification,
                                  self.fragment_offset, self.ttl, self.protocol, self.checksum, self.source_ip, self.destination_ip)

        return ipv4_header


    def unpack(self, buffer):
        ipv4_header_size = struct.calcsize(IPV4_HEAD_FMT)
        ipv4_header_packed = buffer[:ipv4_header_size]
        ipv4_header_unpacked = struct.unpack(IPV4_HEAD_FMT,ipv4_header_packed)
        self.version_ihl = ipv4_header_unpacked[0]
        self.ihl = self.version_ihl & 0xf
        self.version = self.version_ihl >> 4
        self.tos = ipv4_header_unpacked[1]
        self.length = ipv4_header_unpacked[2]
        self.identification = ipv4_header_unpacked[3]
        self.fragment_offset = ipv4_header_unpacked[4]
        self.ttl = ipv4_header_unpacked[5]
        self.protocol = ipv4_header_unpacked[6]
        self.checksum = ipv4_header_unpacked[7]
        self.source_ip = str(ipaddress.IPv4Address(ipv4_header_unpacked[8] ))
        self.destination_ip= str(ipaddress.IPv4Address(ipv4_header_unpacked[9] ))
        self.data = buffer[ipv4_header_size:]

        #print ("source ip  == " + str( ipaddress.IPv4Address(self.source_ip)))
        #print ("destination ip  == " + str( ipaddress.IPv4Address(self.destination_ip)))
        #print ("checksum = "+ str(self.checksum))
        #print ("ttl  == " + str(self.ttl))






