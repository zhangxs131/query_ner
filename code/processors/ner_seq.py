import torch
import os
from torch.utils.data import DataLoader
from util_func.read_file import read_label_list
from processors.bio_dataset import BIO_Dataset


# collate_fn=lambda x: collate_fn(x, info)
def collate_fn(batch, tokenizer, label_list):
    label2id = {k: v for v, k in enumerate(label_list)}
    inputs = tokenizer([list(i['text']) for i in batch], padding="max_length", truncation=True, max_length=512,
                       return_tensors='pt', is_split_into_words=True)

    labels = []
    for id, i in enumerate(batch):
        label = i['label']
        word_ids = inputs.word_ids(batch_index=id)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    inputs['label_ids'] = torch.LongTensor(labels)

    return inputs


def collate_fn_test(batch, tokenizer):
    inputs = tokenizer([list(i) for i in batch], padding="max_length", truncation=True, max_length=512,
                       return_tensors='pt', is_split_into_words=True)
    return inputs


def gen_dataloader(train_data_path,dev_data_path, tokenizer, batch_size=32, label_txt=None):
    if type(label_txt)==str:
        label_txt=read_label_list(label_txt)

    train_dataset = BIO_Dataset(train_data_path)
    dev_dataset = BIO_Dataset(dev_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, label_txt))
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer, label_txt))

    return train_dataloader, dev_dataloader


def gen_test_dataloader(file_dir, tokenizer, batch_size=1):
    if file_dir.split(',')[-1] == 'txt':
        text = read_label_list(file_dir)
        test_loader = DataLoader(text, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda x: collate_fn_test(x, tokenizer=tokenizer))
    else:
        test_loader = None
        print('predict file should be txt line')

    return test_loader