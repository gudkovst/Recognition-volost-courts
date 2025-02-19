import os


def get_liter(name: str) -> str:
    liter = name.split('-')[0]
    return '\U00000462' if liter == "ять" else liter


def make_gt_files(path: str):
    for pict_name in os.listdir(path):
        txt_name = os.path.join(path, pict_name.split('.')[0] + ".gt.txt")
        with open(txt_name, 'w', encoding='utf-8') as file:
            file.write(get_liter(pict_name))


train_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filtered_first\nearest\train"
test_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filtered_first\nearest\test"

make_gt_files(train_path)
make_gt_files(test_path)
