import os
import pandas as pd
from tqdm import tqdm

"""
过滤标注实体前缀多余空格, 后缀空格
过滤量 2597/20w
"""
def filter_pre_space(query,label):
    labels = []
    for j in label:
        if j['text'].startswith(' ') and len(j['text'])>1:
            labels.append(
                {'start': j['start']+1, 'end': j['end'], 'text': j['text'][1:], 'labels': j['labels']})
        elif j['text'].endswith(' ') and len(j['text'])>1:
            labels.append(
                {'start': j['start'], 'end': j['end']-1, 'text': j['text'][:-1], 'labels': j['labels']})
        else:
            labels.append(j)

    return labels

"""
过滤start ids end ids 得到的text 与text 不一致的data
一般都是start ids 定位错误，所以直接进行重新定位处理

过滤数据量 3816/20w
"""
def filter_ids(query,label):

    labels=[]
    for j in label:
        if query[j['start']:j['end']]!=j['text']:
            if query.find(j['text'])<0:
                return False

            labels.append({'start':query.find(j['text']),'end':query.find(j['text'])+len(j['text']),'text':j['text'],'labels':j['labels']})

        else:
            labels.append(j)

    return labels

"""
label 重复的现象
"""
def filter_many(query,label):
    labels = []
    label_dir={}
    for j in label:
        if j['text'] not in label_dir:
            label_dir[j['text']]=j['start']
        else:
            if j['start']==label_dir[j['text']]:
                continue
        labels.append(j)

    return labels

"""
label 不在 设置label_list 中情况
"""
def filter_wrong_label(query,label,label_list):

    label_list=["O",'goods', 'time', 'nonsense', 'price', 'brand', 'ip', 'location', 'company', 'holiday', 'new', 'promotion', 'movie', 'game', 'series', 'height', 'figure', 'suit', 'color-name', 'scene', 'gender', 'crowd', 'color', 'style', 'season', 'material', 'function', 'attribute']


    labels = []
    for j in label:
        if j['labels'] == 'promation':
            labels.append({'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels':'promotion'})
        elif j['labels'] == 'id':
            labels.append({'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels': 'ip'})
        elif j['labels'] not in label_list:
            print(j['labels'])
            return False
        else:
            labels.append(j)

    return labels

"""
过滤无法映射label ，label名字的映射
"""

def filter_pre_label(label):
    filter_attr = ['attr', 'size', 'shade', 'age', 'series','time']
    label_t=[]

    for j in label:
        if j['labels'][0] in filter_attr:
            return False
        elif j['labels'][0] == 'per':
            label_t.append({'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels': 'ip'})
        elif j['labels'][0] == 'loc':
            label_t.append(
                {'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels': 'location'})
        elif j['labels'][0] == 'stopword':
            label_t.append(
                {'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels': 'nonsense'})
        else:
            label_t.append(
                {'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels': j['labels'][0]})

    return label_t


def main():
    data_dir='../data/aug_data/aug_orl_data'
    output_file='../data/tmp_data/pre_ner.csv'
    label_txt = '../data/label_dir/label.txt'
    with open(label_txt, 'r', encoding='utf-8') as f:
        label_name = f.read().splitlines()

    df = pd.DataFrame()
    for filename in os.listdir(data_dir):
        if filename.endswith('.tsv'):
            filepath = os.path.join(data_dir, filename)
            temp = pd.read_csv(filepath,sep='\t')
            df = pd.concat([df, temp], axis=0, ignore_index=True)

    #query去重
    print('合并后 共{}条数据'.format(len(df)))
    df = df.drop_duplicates(subset=['query'])
    print('根据query去重后 {}条数据'.format(len(df)))



    #数据清洗
    result={'query':[],'label':[]}
    for id, data in df.iterrows():
        try:
            label = eval(data['label'])
            label = filter_pre_label(label)
            if not label:
                continue

            label = filter_wrong_label(data['query'], label, label_name)  # 过滤不在
            if not label:
                continue

            label = filter_pre_space(data['query'], label)  # 去掉标注前后缀空格
            label = filter_ids(data['query'], label)  # start end id校准
            label = filter_many(data['query'], label)  # 有些重复识别标注的query进行过滤

            result['label'].append(label)
            result['query'].append(data['query'])
        except Exception as e:
            print(e.args)
            print(data['query'], data['label'])
            continue


    df_result = pd.DataFrame(result)
    print('过滤清洗后共{}条数据集'.format(len(df_result)))
    df_result.to_csv(output_file, index=None)

if __name__ == '__main__':
    main()