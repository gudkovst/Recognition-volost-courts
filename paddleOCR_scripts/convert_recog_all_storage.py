def get_liter(name: str) -> str:
    liter = name.split('-')[0]
    return '\U00000462' if liter == "ять" else liter

def is_first_p(name: str) -> bool:
    return name.split('-')[-1][0] == '9'

def this_liter(liter: str, name: str) -> bool:
    return liter == get_liter(name)

def annotation(name: str, train: bool = True) -> str:
    ds = "train/" if train else "test/"
    liter = get_liter(name)
    return "train_data/rec/" + ds + name + '\t' + liter + '\n'


LENS = {}

def move_and_annotate(*files):
    train_file, test_file = files
    with open(dict_filename, 'r') as dict_file:
        liters = dict_file.read().split('\n')
        picts = list(filter(is_first_p, os.listdir(data_path)))
        for liter in liters:
            liter_picts = list(filter(lambda x: this_liter(liter, x), picts))
            LENS[liter] = len(liter_picts)
            count_train = int(len(liter_picts) * 0.9)
            shuffle(liter_picts)
            for lit in liter_picts[:count_train]:
                os.rename(os.path.join(config.base_dir, liter, lit),
                          os.path.join(config.dest_dir, "train", lit))
                train_file.write(annotation(lit))
            for lit in liter_picts[count_train:]:
                os.rename(os.path.join(config.base_dir, liter, lit),
                          os.path.join(config.dest_dir, "test", lit))
                test_file.write(annotation(lit, False))


if __name__ == "__main__":
    train_file = open(config.train_file_name, 'w', encoding="utf-8")
    test_file  = open(config.test_file_name, 'w', encoding="utf-8")
    move_and_annotate(train_file, test_file)
    train_file.close()
    test_file.close()
