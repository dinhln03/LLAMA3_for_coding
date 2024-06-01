import pdf_to_json as p2j

import json

url = "file:data/multilingual/Latn.SUS/Serif_8/udhr_Latn.SUS_Serif_8.pdf"
lConverter = p2j.pdf_to_json.pdf_to_json_converter()
lConverter.mImageHashOnly = True
lDict = lConverter.convert(url)
print(json.dumps(lDict, indent=4, ensure_ascii=False, sort_keys=True))
