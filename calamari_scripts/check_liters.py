import glob

liters = set()

for name in glob.glob(r"data\test\*.gt.txt"):
    with open(name, 'r', encoding='utf-8') as f:
        s = f.read()
        liters.add(s)
        if s.isupper():
            print(name)

print(liters)
