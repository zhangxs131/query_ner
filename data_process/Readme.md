### 介绍数据处理函数对应的使用场景和功能

1. orl_data_process.py

对标注数据进行合并，去重复，简单的清洗过滤（去除不在设置label中的数据，start end不准确数据纠正。
重复标注的实体，以及实体前后缀多余空格的现象进行修正。

并未对标注数据本身是否正确进行调整。

2. pre_data_process.py 

对之前版本ner数据进行过滤，清洗，标签映射

3. show_conflict_entities.py

为了查看到标注错误的内容，需要查看同一实体被标注为不同label的情况，看是正常不同query多个label
还是标注错误问题。

4. judge_entities_for_sure.py

将3中抽取得到的冲突实体进行分析，按照query数进行排序，当query最高的实体同时大于10时
确定为sure ，其他为unsure

5. ana_entities_labels.py

为了统计所标注的实体各个label所占比重，对3中识别full_entities文件进行识别，绘制柱状图。

6. change_labels.py

解决当时标注时候统一标注策略错误问题，比如中英文品牌词统一识别为一个，等等。

7. gen_BIO_txt.py

将csv数据文件转为符合conll的BIO标注的数据文件
