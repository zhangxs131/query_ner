import pandas as pd
from torch.utils.data import Dataset,DataLoader
from processors.span_dataset import Span_Dataset
from util_func.read_file import read_label_list
import torch


def collate_fn(batch,tokenizer,label_list,max_length=128):

    num_categories=len(label_list)
    label2id={k:v for v,k in enumerate(label_list)}

    inputs=tokenizer([i['text'] for i in batch],padding=True, truncation=True, max_length=128,return_tensors='pt',return_offsets_mapping=True)

    max_seq_len=inputs['input_ids'].shape[1]

    global_label = torch.zeros(
        len(batch),
        num_categories,
        max_seq_len,
        max_seq_len
    )


    for id in range(len(batch)):
        label=batch[id]['label']
        token_mapping = inputs['offset_mapping'][id].numpy()

        start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j[0]!=j[-1]}
        end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j[0]!=j[-1]}

        for info_ in label:
            if info_['start'] in start_mapping and info_['end'] in end_mapping:
                start_idx = start_mapping[info_['start']]
                end_idx = end_mapping[info_['end']]
                if start_idx > end_idx or info_['entity'] == '':
                    continue
                if info_['type'] not in label2id:
                    continue

                global_label[id,label2id[info_['type']], start_idx, end_idx + 1] = 1

    # global_label = global_label.to_sparse()
    inputs.pop('offset_mapping')
    inputs['label_ids']=global_label

    return inputs

def collate_fn_test(batch, tokenizer,max_length=32):
    inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=max_length,
                       return_tensors='pt',return_offsets_mapping=True)
    return inputs


def gen_dataloader(train_data_path,dev_data_path ,tokenizer, batch_size=32, label_txt=None,max_length=32):
    train_dataset = Span_Dataset(train_data_path)
    dev_dataset = Span_Dataset(dev_data_path)

    if type(label_txt)==str:
        label_txt=read_label_list(label_txt)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, label_txt,max_length=max_length))
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer, label_txt,max_length=max_length))

    return train_dataloader, dev_dataloader


def gen_test_dataloader(file_dir, tokenizer, batch_size=1,max_length=32,return_dataset=False):
    if file_dir.split('.')[-1] == 'txt':
        text = read_label_list(file_dir)
        test_loader = DataLoader(text, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda x: collate_fn_test(x, tokenizer=tokenizer,max_length=max_length))
    elif file_dir.endswith('.csv'):
        df=pd.read_csv(file_dir,header=None)
        text=df[1].tolist()
        test_loader = DataLoader(text, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda x: collate_fn_test(x, tokenizer=tokenizer,max_length=max_length))

    if return_dataset:

        return test_loader,text
    else:
        return test_loader


def main():
    train_data = "../../data/train.csv"
    dev_data = "../../data/test.csv"
    label_data = "../../data/new_ner_data/label.txt"
    dev_dataset=Span_Dataset(dev_data)
    from transformers import AutoTokenizer
    label_list=read_label_list(label_data)
    tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')

    dev_dataloader=DataLoader(dev_dataset,batch_size=1,shuffle=False,collate_fn=lambda x:collect_fn(x,tokenizer,label_list))
    for i in dev_dataloader:
        print(i['input_ids'].shape)
        print(i['labels'].shape)
        break



if __name__=="__main__":
    main()