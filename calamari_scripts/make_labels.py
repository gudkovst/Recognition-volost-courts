import os
from utils import make_gt_files


train_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filter_24_03\repeat\train\bicubic"
test_path = r"C:\Users\gudko\history_envs\calamari_p38_env\data\filter_24_03\repeat\test\bicubic"

make_gt_files(train_path)
make_gt_files(test_path)
