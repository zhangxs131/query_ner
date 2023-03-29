import torch
import os
from torch.utils.data import DataLoader
from util_func.read_file import read_label_list
from processors.span_dataset import Span_Dataset


# collate_fn=lambda x: collate_fn(x, info)
def collate_fn(batch, tokenizer, label_list):
    if type(label_list) == str:
        label_list=read_label_list(label_list)

    label2id = {k: v for v, k in enumerate(label_list)}
    inputs = tokenizer([i['text'] for i in batch], padding=True, truncation=True, max_length=512,
                       return_tensors='pt',return_offsets_mapping=True)

    batch_size=len(batch)
    max_seq_length=inputs['input_ids'].shape[1]

    global_label = torch.zeros((
        batch_size,
        max_seq_length,
        max_seq_length), dtype=torch.long
    )

    for id in range(len(batch)):
        label = batch[id]['label']
        token_mapping = inputs['offset_mapping'][id].numpy()

        start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j[0] != j[-1]}
        end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j[0] != j[-1]}

        for info_ in label:
            if info_['start'] in start_mapping and info_['end'] in end_mapping:
                start_idx = start_mapping[info_['start']]
                end_idx = end_mapping[info_['end']]
                if start_idx > end_idx or info_['entity'] == '':
                    continue

                global_label[id, start_idx, end_idx + 1] = label2id[info_['type']]

    inputs.pop('offset_mapping')
    inputs['label_ids'] = global_label

    return inputs


def collate_fn_test(batch, tokenizer):
    inputs = tokenizer([i['text'] for i in batch], padding=True, truncation=True, max_length=512,
                       return_tensors='pt',return_offsets_mapping=True)
    return inputs


def gen_dataloader(train_data_path,dev_data_path ,tokenizer, batch_size=32, label_txt=None):
    train_dataset = Span_Dataset(train_data_path)
    dev_dataset = Span_Dataset(dev_data_path)

    if type(label_txt)==str:
        label_txt=read_label_list(label_txt)


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

if __name__=="__main__":
    train_data="../../data/train.csv"
    dev_data="../../data/test.csv"
    label_data="../../data/new_ner_data/label.txt"
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')
    train_dataloader,dev_dataloade=gen_dataloader(train_data,dev_data,tokenizer,label_txt=label_data)

    for i in train_dataloader:
        print(i['labels'].shape)
