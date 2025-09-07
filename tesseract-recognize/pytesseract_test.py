import pytesseract
import glob
import os
try:
    import Image
except ImportError:
    from PIL import Image


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



data_path = r"C:\Users\User\jupyter\pytesseract_env\completed"


###### https://github.com/h/pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"

path = r"C:\Users\User\jupyter\strings\test"
ext = "*.jpg"

distance = 0
cer = 0
count = 0

for pict in glob.glob(os.path.join(path, ext)):
    img = Image.open(pict)
    res = pytesseract.image_to_string(img, lang='rus')
    if len(res) > 0:
        print(pict, res)
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
img = Image.open(p)
res = pytesseract.image_to_string(img, lang='rus')
print(len(res))
print(levenshtein_distance(res, 'каровскiй Волостной Судъ въ'))

    
