# The MIT License (MIT)
#
# Copyright (c) 2019 Paul Sajna for Adafruit Industries LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
`adafruit_imageload.gif`
====================================================

Load pixel values (indices or colors) into one or more bitmaps and colors into a palette from a GIF file.

* Author(s): Paul Sajna

"""

__version__ = "0.0.0-auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_ImageLoad.git"

bitmaps = []

def load(f):
    bitmaps = []
    palette = []
    table = []
    f.seek(3) 
    version = f.read(3)
    if (version !=  b'89a') and (version != b'87a'):
        raise RuntimeError("Invalid GIF version")
    width = int.from_bytes(f.read(2), 'little')
    height = int.from_bytes(f.read(2), 'little')
    gct_header = int.from_bytes(f.read(1), 'little')
    if (gct_header & 0b10000000) != 0b10000000:
        raise NotImplementedError("Only gifs with a global color table are supported")
    #if (gct_header & 0b0111000 >> 3) + 1 != 8:
        #raise NotImplementedError("Only 8-bit color is supported")
    gct_size = 2 ** ((gct_header & 0b00000111) + 1)
    bg_color_index = int.from_bytes(f.read(1), 'little')
    f.seek(1, 1) # seek one byte relative to the current position (skip a byte)
    for i in range(gct_size):
        color = f.read(3)
        palette.append(color)
    while True:
        separator = f.read(1)
        if separator:
            separator = int.from_bytes(separator, 'little')
            if separator == 0x21:
                # Extension
                label = int.from_bytes(f.read(1), 'little')
                if label == 0xf9:
                    # Graphic Control Extension
                    print("Graphic Control Extension")
                    f.seek(1,1)
                    packed = int.from_bytes(f.read(1), 'little')
                    # delay in seconds between frames
                    delay = int.from_bytes(f.read(2), 'little') / 100
                    # We only care about the transparency flag for now
                    if packed & 1 == 1:
                        transparency_index = int.from_bytes(f.read(1), 'little')
                    else:
                        f.seek(1,1)
                    f.seek(1,1)
                elif label == 0xff:
                    # Application Extension
                    print("Application Extension")
                    f.seek(1,1)
                    application = f.read(8)
                    if application == b'NETSCAPE':
                        f.seek(5,1)
                        loop_count = int.from_bytes(f.read(2), 'little')
                        f.seek(1,1)
                    else:
                        raise NotImplementedError("Unimplemented application extension: " 
                            + ''.join([chr(b) for b in application]))
                elif label == 0xfe:
                    # Comment Extension
                    comment = b''
                    while not comment.endswith(b'\0'):
                        byte = f.read(1)
                        comment += byte
                    comment = ''.join([chr(b) for b in comment])
                    print(comment)
                else:
                    raise NotImplementedError("Unimplemented extension: " + hex(label))
            elif separator == 0x2c:
                # Image Descriptor
                print("Image Descriptor")
                image_start_x = int.from_bytes(f.read(2), 'little')
                image_start_y = int.from_bytes(f.read(2), 'little')
                image_width = int.from_bytes(f.read(2), 'little')
                image_height = int.from_bytes(f.read(2), 'little')
                # Ignore the packed fields for now
                f.seek(1,1)
                # Image Data
                print("Image Data")
                lzw_code_size = int.from_bytes(f.read(1), 'little')
                compressed = bytearray()
                while True:
                    block_size = int.from_bytes(f.read(1), 'little')
                    if block_size == 0:
                        break
                    compressed += f.read(block_size)
                bitmap = decompress(compressed, lzw_code_size)
                bitmaps.append(bitmap)
            elif separator == 0x3b:
                # Trailer
                break
            else:
                raise RuntimeError("Got an unexpected separator: " + hex(separator))

def decompress(block, min_code_size):
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    cur_code_size = min_code_size + 1
    bit_offset = 0
    code_stream = []
    index_stream = []
    table = []
    prev_code = None
    nextcode = clear_code + 2
    while bit_offset < 8*(len(block)-1):
        if nextcode == (1 << cur_code_size):
            cur_code_size += 1
        code = fetch_bits(block, cur_code_size, bit_offset)
        #print(code, prev_code)
        bit_offset += cur_code_size
        if code == clear_code:
        #    print(table)
        #    print(len(table)) 
            table = [[i] for i in range(1 << min_code_size)]
            table.append([clear_code])
            table.append([eoi_code])
        #    print(table)
            nextcode = clear_code + 2
            prev_code = None
            print("table reset")
            continue
        elif code == eoi_code:
            print("stop")
            break
        elif code < len(table):
            index_stream.append(table[code])
            k = [table[code][0]]
            if prev_code is not None:
                table.append(table[prev_code] + k)
                nextcode +=1
        elif prev_code is None:
            raise ValueError("First code after a reset must be in the table")
        else:
            k = [table[prev_code][0]]
            index_stream.append(table[prev_code] + k)
            table.append(table[prev_code] + k)
            nextcode +=1
        prev_code = code
        #nextcode = len(table)
    index_stream = flatten(index_stream)
    #print(index_stream)
    return index_stream

def fetch_bits(bytearr, nbits, bit_offset):
    byte_offset = bit_offset//8
    rem = bit_offset % 8
    bits = 0
    for i in range(nbits):
        bit = (bytearr[byte_offset] | (bytearr[byte_offset+1] << 8)) & (1 << (rem + i))
        bits |= bit >> (rem)
    return bits
    

def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items
