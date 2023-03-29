from torch.utils.data import Dataset
import json


class BIO_Dataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data = self._load_file(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _load_file(self, file_name):
        data = []

        if file_name.split('.')[-1] == 'json':
            with open(file_name, encoding='utf-8') as f:
                content = f.readlines()
            for i in content:
                data.append(json.loads(i))
        elif file_name.split('.')[-1] == 'txt':
            with open(file_name, encoding='utf-8') as f:
                content = f.readlines()
            text = []
            label = []
            for i in content:
                if i == '\n':
                    data.append({'text': text, 'label': label})
                    text = []
                    label = []
                else:
                    t = i.split(' ')
                    if len(t) == 2:
                        text.append(t[0])
                        label.append(t[1].strip('\n'))
                    else:
                        text.append(' ')
                        label.append(t[0].strip('\n'))

        return data  # text,label