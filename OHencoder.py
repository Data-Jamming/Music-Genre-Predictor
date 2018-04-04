def encodeAll(lyrics_set):
    pass

#takes a 2d list of elements and returns a dictionary that takes the element and gives 
#you the One Hot Encoded Vector's index
#Threshold is how many occurrences are needed before it's given a spot in the vector
def map_to_int_ids(data, threshold=1):
    """
    e.g. ["word", "another", "word", "end"] => {"word": 0, "another": 1, "end": 2}
    Params:
        data (list): strings.
            e.g. ["Rock", "Pop", "Jazz"]
            e.g. ["Sing", "me", "a", "song", "you're", "the", "piano", "man"]

    NOTE: case insensitive (e.g. "this" and "This" are treated as equivalent).
    """
    dic_all = {}
    dic = {}
    for elements in data:
        for element in elements:
            element = element.lower()
            if element not in dic:
                t = {element:0}
                dic.update(t)
            if element not in dic_all:
                t = {element:0}
                dic_all.update(t)
        for x in dic:
            x = x.lower()
            dic_all[x] += 1
        dic = {}
        
    dic = {}    
    
    i = 0
    for element in dic_all:
        if dic_all[element] >= threshold:
            t = {element:i}
            dic.update(t)
            i += 1
    return dic

