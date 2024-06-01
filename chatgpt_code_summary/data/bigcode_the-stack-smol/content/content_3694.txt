# Get substring using 'start' and 'end' position.
def get_substring_or_empty(data, start, end=''):
    if start in data:
        if '' == start:
            f = 0
        else:
            f = len(start)
            f = data.find(start) + f
        data = data[f:]
    else:
        return ''

    if end in data:
        if '' == end:
            f = len(data)
        else:
            f = data.find(end)
        data = data[:f]
    else:
        return ''

    data = data.strip()
    return data

