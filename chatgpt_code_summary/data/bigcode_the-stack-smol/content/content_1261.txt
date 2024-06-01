import json

def get_text_from_json(file):

    with open(file) as f:
        json_text = f.read()
        text = json.loads(json_text)
        return text

content = get_text_from_json('content.json')

test = get_text_from_json('test.json')