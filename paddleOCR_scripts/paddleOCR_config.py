from os.path import join as ospj

base_dir = r"C:\Users\User\jupyter\paddleOCR_env\train_data\completed"
dest_dir = r"C:\Users\User\jupyter\paddleOCR_train_env\train_data\rec"

train_file_name = ospj(dest_dir, "rec_gt_train.txt")
test_file_name  = ospj(dest_dir, "rec_gt_test.txt")
dict_file_name  = ospj(dest_dir, "volost_dict_v0.txt")

test_img_path = ospj(dest_dir, "test")
