import paddleOCR_config as config
import os


def get_liter(name: str) -> str:
    liter = name.split('-')[0]
    return '\U00000462' if liter == "ять" else liter


liter_picts = os.listdir(os.path.join(config.dest_dir, "test"))
liter_set = {get_liter(l) for l in liter_picts}
with open(config.dict_file_name, 'w', encoding="utf-8") as df:
    for l in liter_set:
        print(l, file=df, end="\n")
