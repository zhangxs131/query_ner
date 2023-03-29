import pandas as pd
import re
import sys


def has_chinese_english(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]+[a-zA-Z]+|[a-zA-Z]+[\u4e00-\u9fa5]+')  # 匹配同时包含中英文的字符串
    return True if pattern.fullmatch(text) else False


def show_example(file_name):
    df = pd.read_csv(file_name)
    q=[]
    l=[]

    wrong_data = 0

    for i, data in df.iterrows():
        query = data['query']
        if has_chinese_english(query):
            label = eval(data['label'])

            for j in label:
                if len(j['text']) < 5:
                    continue
                if j['labels'] != 'brand':
                    continue
                if has_chinese_english(j['text']):
                    print(query, label)
                    wrong_data += 1
                    break
                q.append(data['query'])
                l.append(data['label'])

    print(wrong_data)

"""
中英文brand 整体切分

过滤数据量 322/20w
"""
def filter_zhen(query,label):

    def has_chinese_english(text):
        pattern = re.compile(r'[\u4e00-\u9fa5]+[a-zA-Z]+|[a-zA-Z]+[\u4e00-\u9fa5]+')  # 匹配同时包含中英文的字符串
        return True if pattern.fullmatch(text) else False

    def split_string(text):
        pattern = re.compile(r'[\u4e00-\u9fa5]+|[a-zA-Z]+')  # 匹配中文或英文，进行切分
        return pattern.findall(text)

    if not has_chinese_english(query):
        return label

    labels=[]
    for j in label:
        if len(j['text'])>5 and j['labels'] == 'brand' and has_chinese_english(j['text']):
            print(label)

            sub_str=split_string(j['text'])
            for st in sub_str:
                labels.append({'start':query.find(st),'end':query.find(st)+len(st),'text':st,'labels':j['labels']})
            print(labels)
        else:
            labels.append(j)

    return labels



def main():

    output_file= '../data/tmp_data/filtered_enzh.csv'
    file_name='../data/tmp_data/merged.csv'

    #数据清洗
    df = pd.read_csv(file_name)
    #show_example(file_name)

    result={'query':[],'first_cate':[],'second_cate':[],'label':[]}
    for id,data in df.iterrows():
        try:
            label=eval(data['label'])
            label = filter_zhen(data['query'], label)  # 中英文brand（len>5)切分，后面可能需要实体统一性校对，比如xx女装 分开后女装被标注为brand

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