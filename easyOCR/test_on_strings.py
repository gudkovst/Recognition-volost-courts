import easyocr
import glob
import os

def levenshtein_distance(s1, s2, damerau=False):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1
 
    for i in range(lenstr1):
        for j in range(lenstr2):
            cost = int(s1[i] != s2[j])
            d[(i, j)] = min(d[(i-1, j)] + 1,      # deletion
                            d[(i, j-1)] + 1,      # insertion
                            d[(i-1, j-1)] + cost) # substitution
            if damerau and i and j and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i-2, j-2] + 1) # transposition
 
    return d[lenstr1 - 1, lenstr2 - 1]


reader = easyocr.Reader(['ru']) # this needs to run only once to load the model into memory

path = r"C:\Users\User\jupyter\strings\test"
ext = "*.jpg"

distance = 0
cer = 0
count = 0

for pict in glob.glob(os.path.join(path, ext)):
    result = reader.readtext(pict, detail=0)
    res = ' '.join(result)
    name = os.path.splitext(os.path.basename(pict))[0]
    name += '.gt.txt'
    with open(os.path.join(path, name), encoding='utf-8') as gt:
        true_rec = gt.read()
    d = levenshtein_distance(res, true_rec)
    distance += d
    cer += d / len(true_rec)
    count += 1

print('average dist:', distance / count)
print('average CER:', cer / count)

p = r"C:\Users\User\jupyter\strings\test\134538-937.jpg"
result = reader.readtext(p, detail=0)
s = ' '.join(result)
print(levenshtein_distance(s, 'каровскiй Волостной Судъ въ'))

    
