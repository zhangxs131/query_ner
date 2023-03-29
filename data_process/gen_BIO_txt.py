import sys
import pandas as pd
from tqdm import tqdm

def gen_BIO(_line, labels):
    label = len(_line) * ['O']
    for _preditc in labels:


        label[_preditc['start']] = 'B-' + _preditc['labels']
        label[_preditc['start'] + 1: _preditc['end']] = (_preditc['end'] - _preditc[
            'start'] - 1) * [('I-' + _preditc['labels'])]

    return label

def main():

    file_name='../data/tmp_data/服饰鞋包.csv'
    save_name='bio_data_20w.txt'
    if len(sys.argv)>1:
        file_name=sys.argv[1]
    if len(sys.argv)>2:
        save_name=sys.argv[2]


    df=pd.read_csv(file_name)

    with open(save_name,'w',encoding='utf-8') as f:
        for i, data in df.iterrows():
            try:
                data['label'] = eval(data['label'])
                label=gen_BIO(data['query'],data['label'])
                assert len(label)==len(data['query'])
                data['query']=list(data['query'])
                for j in range(len(label)):
                    f.write('{} {}\n'.format(data['query'][j],label[j]))
                f.write('\n')
            except:
                print(label, data['query'])
                continue

if __name__=='__main__':
    main()