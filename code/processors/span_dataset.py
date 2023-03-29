from torch.utils.data import Dataset
import json
import pandas as pd


class Span_Dataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data = self._load_file(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _load_file(self, file_name):
        data_list = []

        df = pd.read_csv(file_name)

        # 生成datalist 'text', 'label:[{start_idx,end_idx,type,entity} ]

        query = df['query'].to_list()
        label = df['label'].to_list()

        for q, l in zip(query, label):
            t = {}
            t['text'] = q
            t['label'] = []
            try:
                if type(l) == str:
                    l=eval(l)

                for j in l:
                    t['label'].append({'start': j['start'], 'end': j['end'], 'type': j['labels'],
                                           'entity': j['text']})
                data_list.append(t)
            except:
                print(l)

        return data_list  # text,label

if __name__=="__main__":
    file_name="../../data/test.csv"
    train_dataset=Span_Dataset(file_name)
    print(train_dataset[0])