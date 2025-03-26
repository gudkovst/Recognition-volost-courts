def get_liter(name: str) -> str:
    union_liters = {"ф+большая": 'ф', "д+верхнее": 'd', "л+длинное": 'l', "т+молотком": 't'}
    liter = name.replace('-', '+', 1).split('-')[0]
    if liter.lower() in union_liters:
        lit = union_liters[liter.lower()]
    elif "большая" in liter:
        lit = liter[0].upper()
    elif "ять" in liter:
        lit = '\U00000462'.lower()
    else:
        lit = liter[0].lower()
    return lit.replace(' ', '-')


def make_gt_files(path: str):
    for pict_name in os.listdir(path):
        txt_name = os.path.join(path, pict_name.split('.')[0] + ".gt.txt")
        with open(txt_name, 'w', encoding='utf-8') as file:
            file.write(get_liter(pict_name))
