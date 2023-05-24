# query-ner 项目代码

本项目用于电商平台搜索query的ner识别项目，也可用于其他query，商品spu-name等文本ner任务任务中。支持常见的ner框架：

- bert+softmax
- bert+crf
- bert+span
- bert+baffine
- bert+gp

以及bert类预训练模型，可替换为多种中文bert类预训练模型。其中bert+gp支持嵌套ner的识别。

Requirement:
======

	Python: 3.7   
	numpy
	pandas
	pytorch
	transformers

Input format:
======

项目中run_ner_crf.py 使用BIO格式ner数据输入

	美	B-LOC
	国	E-LOC
	的	O
	华	B-PER
	莱	I-PER
	士	E-PER
	
	我	O
	跟	O
	他	O
	谈	O
	笑	O
	风	O
	生	O 

其他span,baffine,gp框架下，使用以下 span格式csv文件作为输入

```
query,label

男裤阔腿,"[{'start': 0, 'end': 2, 'text': '男裤', 'labels': 'goods'}, {'start': 2, 'end': 4, 'text': '阔腿', 'labels': 'attribute'}]"
胖男生冬季棉服,"[{'start': 0, 'end': 1, 'text': '胖', 'labels': 'figure'}, {'start': 1, 'end': 3, 'text': '男生', 'labels': 'gender'}, {'start': 3, 'end': 5, 'text': '冬季', 'labels': 'season'}, {'start': 5, 'end': 7, 'text': '棉服', 'labels': 'goods'}]"
2023裙子新年,"[{'start': 0, 'end': 4, 'text': '2023', 'labels': 'time'}, {'start': 4, 'end': 6, 'text': '裙子', 'labels': 'goods'}, {'start': 6, 'end': 8, 'text': '新年', 'labels': 'holiday'}]"
过生日穿的裙子冬天,"[{'start': 0, 'end': 3, 'text': '过生日', 'labels': 'scene'}, {'start': 3, 'end': 5, 'text': '穿的', 'labels': 'nonsense'}, {'start': 5, 'end': 7, 'text': '裙子', 'labels': 'goods'}, {'start': 7, 'end': 9, 'text': '冬天', 'labels': 'season'}]"
牛仔裤阔腿秋冬,"[{'start': 0, 'end': 3, 'text': '牛仔裤', 'labels': 'goods'}, {'start': 3, 'end': 5, 'text': '阔腿', 'labels': 'attribute'}, {'start': 5, 'end': 7, 'text': '秋冬', 'labels': 'season'}]"
男生的羽绒服,"[{'start': 0, 'end': 2, 'text': '男生', 'labels': 'gender'}, {'start': 2, 'end': 3, 'text': '的', 'labels': 'nonsense'}, {'start': 3, 'end': 6, 'text': '羽绒服', 'labels': 'goods'}]"
```



How to run the code?
====

#### train：

1. 将训练集，验证集以及ner的label文件存入 data中。

2. 将预训练模型保存入，pretrain_model中。

3. 修改code/script中对应框架下的shell文件

   如 run_ner_gp.sh

   ```sh
   CURRENT_DIR=`pwd`
   export BERT_BASE_DIR=../pretrain_model/roberta-wwm-chinese
   export OUTPUR_DIR=../outputs/gp
   TASK_NAME="queryner"
   #
   python run_ner_gp.py \
     --model_type=bert \
     --train_data_path ../data/test_data/train.csv \
     --dev_data_path ../data/test_data/dev.csv \
     --label_txt ../data/label_dir/label.txt \
     --model_name_or_path=$BERT_BASE_DIR \
     --task_name=$TASK_NAME \
     --do_eval \
     --do_lower_case \
     --train_max_seq_length=128 \
     --eval_max_seq_length=512 \
     --per_gpu_train_batch_size=24 \
     --per_gpu_eval_batch_size=24 \
     --learning_rate=3e-5 \
     --crf_learning_rate=1e-3 \
     --num_train_epochs=4.0 \
     --logging_steps=-1 \
     --save_steps=-1 \
     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
     --overwrite_output_dir \
     --seed=42
   ```

4. 运行训练脚本

   ```sh
   sh script/run_ner_gp.sh
   ```

5. 参数设置
   - model_name_or_path 预训练模型地址
   - do_adv 使用fgm进行对抗训练
   - label_txt 标签txt文件地址
   - train_data_path 训练数据地址
   - dev_data_path 验证集地址

#### Predict:

1. 运行sh脚本，如：predict_ner_gp.sh

   ```shell
   export BERT_BASE_DIR=../outputs/gp/queryner_output_0515_full/bert/checkpoint-38685
   export OUTPUR_DIR=../outputs/gp
   TASK_NAME="queryner"
   #
   python run_ner_gp.py \
     --model_type=bert \
     --predict_data_path ../data/red_spu_left.csv  \
     --result_data_path ../data/0523_span.csv \
     --save_type span_csv \
     --label_txt ../data/label_dir/label_p0.txt \
     --model_name_or_path=$BERT_BASE_DIR \
     --task_name=$TASK_NAME \
     --do_predict \
     --do_lower_case \
     --eval_max_seq_length 32 \
     --per_gpu_eval_batch_size 256 \
     --save_steps=-1 \
     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
     --overwrite_output_dir \
     --seed=42
   
   ```

2. 参数设置

   - predict_data_path 待预测文件，可以为csv文件（query 列为待预测text）或txt文件
   - save_type可选（span_csv,span_json,bio_csv,bio_txt) 4种类型作为结果保存文件。
   - eval_max_seq_length 分词的max_length参数，根据输入文本确定，影响预测速度。



## Acknowledge: 

参考网上开源项目：

- [lonePatient](https://github.com/lonePatient)/**[BERT-NER-Pytorch](https://github.com/lonePatient/BERT-NER-Pytorch)** 
- [xiangking](https://github.com/xiangking)/**[ark-nlp](https://github.com/xiangking/ark-nlp)** 



