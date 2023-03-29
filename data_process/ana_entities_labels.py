import json
import sys
import matplotlib.pyplot as plt
import pandas as pd

def draw_histogram(title_list,value_list,save_name=None):
    # 创建一个图形对象和一个子图对象
    fig, ax = plt.subplots()

    # 绘制柱状图
    bars = ax.bar(title_list, value_list,width=0.3,align='edge')

    # 设置标题和标签
    ax.set_title('Label anaylse',fontsize=16)
    ax.set_xlabel('Label',fontsize=12)
    ax.set_ylabel('nums',fontsize=12)

    # 调整x轴标签的字体大小
    ax.tick_params(axis='x', labelsize=8)

    # 在每个柱状图柱子的顶部显示具体的value值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

    # 显示图形
    plt.show()


def ana_labels(file_name,save_name):
    with open(file_name,'r',encoding='utf-8') as f:
        orl_dict=json.loads(f.read())

    # label dict生成
    label_txt = '../data/label_dir/label.txt'
    label_cn_txt = '../data/label_dir/label_cn.txt'
    with open(label_txt, 'r', encoding='utf-8') as f:
        label_name = f.read().splitlines()

    label_nums_dir={}

    for k,v in orl_dict.items():
        for k1,v1 in v.items():
            if k1 not in  label_nums_dir:
                label_nums_dir[k1]=0
            label_nums_dir[k1]+=v1['nums']

    # 标题列表和值列表
    value_list = [label_nums_dir[i] if i in label_nums_dir else 0 for i in label_name]

    draw_histogram(label_name[1:],value_list[1:],save_name)

def main():
    file1 = '../data/tmp_data/full_entities.json'
    save_name='../data/tmp_data/analyse_labels.png'
    if len(sys.argv) > 1:
        file1 = sys.argv[1]
    if len(sys.argv) > 2:
        save_name= sys.argv[2]

    ana_labels(file1,save_name)

if __name__=='__main__':
    main()
