def read_label_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read().splitlines()

    return content