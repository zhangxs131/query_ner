import pandas as pd
import sys
import json
from tqdm import tqdm
import os


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


def main():
    # label dict生成
    label_txt = '../data/label_dir/label.txt'
    label_cn_txt = '../data/label_dir/label_cn.txt'
    with open(label_txt, 'r', encoding='utf-8') as f:
        label_name = f.read().splitlines()

    with open(label_cn_txt, 'r', encoding='utf-8') as f:
        label_cn_name = f.read().splitlines()
    name_dir = {}
    for l_name, l_cn_name in zip(label_name, label_cn_name):
        name_dir[l_cn_name] = l_name


    orl_data_dir = '../data/orl_data/'
    output_file = "orl_merged.csv"  # 合并后的CSV文件的文件名

    # 遍历文件夹中的CSV文件，将它们合并到一个DataFrame中
    df = pd.DataFrame()
    for filename in os.listdir(orl_data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(orl_data_dir, filename)
            temp = pd.read_csv(filepath)
            df = pd.concat([df, temp], axis=0, ignore_index=True)

    #query去重
    print('合并后 共{}条数据'.format(len(df)))
    df = df.drop_duplicates(subset=['query'])
    print('根据query去重后 {}条数据'.format(len(df)))



    #数据清洗
    result={'query':[],'first_cate':[],'second_cate':[],'label':[]}
    for id,data in df.iterrows():
        try:
            if not data['label'].startswith('['):
                t = data['label'].split('}{')
                t = ('},{').join(t)
                data['label'] = "[" + t + "]"
            label=eval(data['label'])
            label_t=[]
            for j in label:
                label_t.append({'start': j['start'], 'end': j['end'], 'text': j['text'], 'labels':name_dir[j['labels'][0]]})
            label=label_t

            label = filter_wrong_label(data['query'], label, label_name)  # 过滤不在
            if not label:
                continue

            label = filter_pre_space(data['query'], label)  # 去掉标注前后缀空格
            label = filter_ids(data['query'], label)  # start end id校准
            label = filter_many(data['query'], label)  # 有些重复识别标注的query进行过滤

            result['label'].append(label)
            result['query'].append(data['query'])
            result['first_cate'].append(data['first_cate'])
            result['second_cate'].append(data['second_cate'])
        except Exception as e:
            print(e.args)
            print(data['query'],data['label'])
            continue

    df_result = pd.DataFrame(result)
    print('过滤清洗后共{}条数据集'.format(len(df_result)))
    df_result.to_csv(output_file, index=None)



if __name__ == '__main__':
    main()
