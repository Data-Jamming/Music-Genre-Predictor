
#takes a list of elements and returns a dictionary that takes the element and gives you the One Hot Encoded Vector's index
def encode(data):
    dic = {}
    i = 0
    for element in data:
        if element not in dic:
            t = {element:i}
            i += 1
            dic.update(t)
    return dic