def encodeAll(lyrics_set):
    pass

def map_to_int_ids(data):
    """
    e.g. ["word", "another", "word", "end"] => {"word": 0, "another": 1, "end": 2}
    Params:
        data (list): strings.
            e.g. ["Rock", "Pop", "Jazz"]
            e.g. ["Sing", "me", "a", "song", "you're", "the", "piano", "man"]

    NOTE: case insensitive (e.g. "this" and "This" are treated as equivalent).
    """
    dic = dict()
    i = 0
    for element in data:
        element = element.lower()
        if element not in dic:
            t = {element:i}
            i += 1
            dic.update(t)
    return dic