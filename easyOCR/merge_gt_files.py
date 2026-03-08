import os

def merge_text_files(input_dir, output_file):
    """
    Считывает данные из всех текстовых файлов директории и записывает объединённую строку в выходной файл.
    
    Args:
        input_dir (str): Путь к директории с текстовыми файлами
        output_file (str): Путь к выходному файлу
    """
    all_text = []
    if not os.path.isdir(input_dir):
        raise ValueError(f"Директория '{input_dir}' не существует или не является директорией")

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_text.append(content)
            except UnicodeDecodeError:
                print(f"Пропущен файл {filename} (не текстовый формат)")
            except Exception as e:
                print(f"Ошибка при чтении файла {filename}: {e}")
    
    combined_text = '\n'.join(all_text)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)


if __name__ == "__main__":
    input_dir = r"C:\Users\gudko\history_envs\easyocr_env\predictGen\str"
    output_file = r"C:\Users\gudko\history_envs\easyocr_env\predictGen\str.txt"
    merge_text_files(input_dir, output_file)
