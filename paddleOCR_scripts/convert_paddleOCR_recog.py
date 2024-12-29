import os
from random import shuffle
import paddleOCR_config as config


def annotation(name: str, train: bool = True) -> str:
    ds = "train/" if train else "test/"
    liter = name.split('-')[0]
    liter = '\U00000462' if liter == "ять" else liter
    return "train_data/rec/" + ds + name + '\t' + liter + '\n'


def move_and_annotate(files: list):
    train_file, test_file = files
    for liter_storage in os.listdir(config.base_dir):
        liter_dir = os.path.join(config.base_dir, liter_storage)
        liters = os.listdir(liter_dir)
        count_train = int(len(liters) * 0.9)
        shuffle(liters)
        for liter in liters[:count_train]:
            os.rename(os.path.join(config.base_dir, liter_storage, liter),
                      os.path.join(config.dest_dir, "train", liter))
            train_file.write(annotation(liter))
        for liter in liters[count_train:]:
            os.rename(os.path.join(config.base_dir, liter_storage, liter),
                      os.path.join(config.dest_dir, "test", liter))
            test_file.write(annotation(liter, False))


def annotate(files: list):
    def __annotate(file, dirc):
        liters = os.listdir(dirc)
        shuffle(liters)
        train = 'test' not in file.name
        for liter in liters:
            file.write(annotation(liter, train))
    
    train_file, test_file = files
    train_dir = os.path.join(config.dest_dir, "train")
    __annotate(train_file, train_dir)
    test_dir = os.path.join(config.dest_dir, "test")
    __annotate(test_file, test_dir)
    

if __name__ == "__main__":
    train_file = open(config.train_file_name, 'w', encoding="utf-8")
    test_file  = open(config.test_file_name, 'w', encoding="utf-8")
    annotate([train_file, test_file])
    train_file.close()
    test_file.close()
