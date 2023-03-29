import pandas as pd
import sys
import json
from tqdm import tqdm

def gen_entities(file_name,save_name,full_entities_outfile=None):
    df=pd.read_csv(file_name)
    entities={}
    query_example=3

    for i,data in tqdm(df.iterrows()):
        query=data['query']
        label=eval(data['label'])

        for j in label:
            if j['text'] not in entities:
                entities[j['text']]={j['labels']:{'querys':[query],'nums':1}}
            else:
                if j['labels'] in entities[j['text']]:
                    if entities[j['text']][j['labels']]['nums']<query_example:
                        entities[j['text']][j['labels']]['querys'].append(query)
                        entities[j['text']][j['labels']]['nums']+=1
                    else:
                        entities[j['text']][j['labels']]['nums'] += 1
                else:
                    entities[j['text']][j['labels']]={'querys':[query],'nums':1}

    print('全部实体共{}个'.format(len(entities)))
    if full_entities_outfile:
        with open(full_entities_outfile, "w") as outfile:
            json.dump(entities, outfile, indent=4,ensure_ascii=False)


    result={}
    for k,v in entities.items():
        if len(v)>=2 and len(k)>=2:
            result[k]=v

    print('有冲突的实体有 {} 个'.format(len(result)))

    with open(save_name, "w") as outfile:
        json.dump(result, outfile, indent=4,ensure_ascii=False)



def main():
    file1 = '../data/tmp_data/merged.csv'
    save_name = '../data/tmp_data/conflict_entities.json'
    full_entities_name = '../data/tmp_data/full_entities.json'
    if len(sys.argv) > 1:
        file1 = sys.argv[1]
    if len(sys.argv) > 2:
        save_name = sys.argv[2]
    gen_entities(file1,save_name,full_entities_name)

if __name__=='__main__':
    main()
