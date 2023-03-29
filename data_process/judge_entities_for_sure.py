import json
import sys

import pandas as pd


def filter_sure_json(file_name,sure_name,unsure_name):
    with open(file_name,'r',encoding='utf-8') as f:
        orl_dict=json.loads(f.read())

    sure_dir={}
    unsure_dir={}

    for k,v in orl_dict.items():
        r=True
        # for k1,v1 in v.items():
        # if 'goods' in v:
        #     continue
        #
        # if 'brand' in v:
        #     sure_dir[k]={'brand':v['brand']}
        #     continue

        for k1,v1 in sorted(v.items(), key=lambda kv:(kv[1]['nums']),reverse=True):
            if v1['nums']>10:
                # if k not in sure_dir or sure_dir[k][sure_dir[k].keys()[0]]['nums']<v1['nums']:
                sure_dir[k]={k1:v1}
                r=False
                break

        if r:
            unsure_dir[k]=v
    print('原冲突实体有 {} 个'.format(len(orl_dict)))
    print('解决冲突的实体有 {} 个'.format(len(sure_dir)))
    print('仍然有冲突的实体有 {} 个'.format(len(unsure_dir)))

    with open(sure_name, "w") as outfile:
        json.dump(sure_dir, outfile, indent=4, ensure_ascii=False)

    entities=[]
    labels=[]
    for k,v in unsure_dir.items():
        entities.append(k)
        labels.append(v)

    df=pd.DataFrame({'实体':entities,'冲突标签':labels})
    df.to_csv('unsure.csv',index=None)
    # with open(unsure_name, "w") as outfile:
    #     json.dump(unsure_dir, outfile, indent=4, ensure_ascii=False)


def main():
    file1 = '../data/tmp_data/conflict_entities.json'
    sure_name = '../data/tmp_data/sure.json'
    unsure_name = '../data/tmp_data/unsure.json'
    if len(sys.argv) > 1:
        file1 = sys.argv[1]
    if len(sys.argv) > 2:
        sure_name = sys.argv[2]
    if len(sys.argv) > 3:
        unsure_name = sys.argv[3]

    filter_sure_json(file1,sure_name,unsure_name)

if __name__=='__main__':
    main()
