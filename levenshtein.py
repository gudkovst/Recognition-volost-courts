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


actual = '''Крестьянка с. Тулинскаго
Александра Самойлова
Бокарева проситъ о ис
ключенiи изъ описи иму
щества у мужа ее Фе
дора Бокарева: домъ, ам
баръ и 5 сосновыхъ бревенъ
какъ принадлежащего ей
Александрѣ Бокаре
вой'''

pred = '''Крестьянка с. Тулинскаго
Александра Симойлова
Бокарева просить о
ключении из обниси иму
щества у мурна ег сре¬
ддана вокарева дом, или
барь и 5 сосновых древени
какъ принидилкащаго
его Александре Бакири
егой'''

print(f"levenshtein_distance: {levenshtein_distance(actual, pred)}")
print(f"normalized levenshtein_distance: {levenshtein_distance(actual, pred) / len(actual)}")
print(f"max normalized levenshtein_distance: {levenshtein_distance(actual, pred)/ max(len(actual), len(pred))}")
