#!/usr/bin/env python3

# Data schema:
# (start) (12b junk) artist (5* byte) (1b junk) title (col) (1b junk) date and time (col) (1b junk) url (urldur) duration (col) (1b junk) thumbnail url (end)

keybytes = {
    "row_start": "80 09 80 00 80", # row start
    "col": "5F 10", # column delimeter
    "urldur": "58", # url/duration delimeter
    "urldur2": "D8",
    "urldur3": "D2",
    "row_end": "D8 00 0A 00 2A 00 2B 00 2C 00 2D 00 2E 00 2F 00 30 00 31 00 32 00" # row end
}

# convert hex to bytes
for k, v in keybytes.items():
    keybytes[k] = bytearray.fromhex(v)

def get_urls_from_playlist(filename):
    with open(filename, "rb") as f:
        content = f.read()
        for row in content.split(keybytes["row_start"])[1:]:
            try:
                row = row.split(keybytes["row_end"])[0] # cut off everything after the row end
                columns = row.split(keybytes["col"])
                for col in columns:
                    if "http" in str(col):
                        # cut off junk bytes
                        url = col.split(keybytes["urldur"])[0].split(keybytes["urldur2"])[0].split(keybytes["urldur3"])[0]
                        url = url[1:].decode("utf-8")
                        yield url
            except Exception as e:
                pass
