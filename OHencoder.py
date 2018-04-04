
#takes a 2d list of elements and returns a dictionary that takes the element and gives 
#you the One Hot Encoded Vector's index
#Threshold is how many occurrences are needed before it's given a spot in the vector
def encode(data, threshold = 1):

    dic_all = {}
    dic = {}
    for elements in data:
        for element in elements:
            if element not in dic:
                t = {element:0}
                dic.update(t)
            if element not in dic_all:
                t = {element:0}
                dic_all.update(t)
        for x in dic:
            dic_all[x] += 1
        dic = {}
        
    i = 0
    for element in dic_all:
        dic_all[element] = i
        i += 1
    return dic_all 
    #haha and then it returns dick-all I didn't even do that on purpose