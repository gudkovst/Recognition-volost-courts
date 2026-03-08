import os.path

import cv2
import glob
import subprocess
from test_detect_string import main_detect_save
from line_handler import handler_line
from merge_gt_files import merge_text_files


def evaluate_model(model_checkpoints_path, data_path, predict_files_path):
    data_path_cmd = os.path.join(data_path, "*.jpg")
    cmd = ["calamari-predict",  "--checkpoint",  model_checkpoints_path,
           "--files", data_path_cmd, "--output_dir", predict_files_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    model = r"C:\Users\gudko\history_envs\calamari_p38_env\string_model\long_first_20str\best.ckpt"
    data = r"C:\Users\gudko\history_envs\easyocr_env\rois"
    save = r"C:\Users\gudko\history_envs\easyocr_env\predictGen\test"
    evaluate_model(model, data, save)
    output_file = r"C:\Users\gudko\history_envs\easyocr_env\predictGen\test.txt"
    merge_text_files(save, output_file)


if __name__ == "__n__":
    im_path = r"C:\Users\gudko\history_envs\easyocr_env\big_block.jpg"
    save_path = r"C:\Users\gudko\history_envs\easyocr_env\big_block"
    visual_save_path = r"C:\Users\gudko\history_envs\easyocr_env\lines_detected.jpg"
    main_detect_save(im_path, save_path, visual_save_path)

    for i, im in enumerate(glob.glob(os.path.join(save_path, '*.jpg'))):
        im_path = os.path.join(save_path, im)
        save_frag_path = os.path.join(save_path, f"line_{i + 100}_roi.jpg")
        handler_line(im_path, save_frag_path, 32)

    input_dir = r"C:\Users\gudko\history_envs\easyocr_env\predictGen\str"
    output_file = r"C:\Users\gudko\history_envs\easyocr_env\predictGen\str.txt"

    # TODO: add call for models

    merge_text_files(input_dir, output_file)
