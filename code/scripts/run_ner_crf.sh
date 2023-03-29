CURRENT_DIR=`pwd`
export BERT_BASE_DIR=../pretrain_model/bert-base-chinese
export OUTPUR_DIR=../outputs/crf/
TASK_NAME="queryner"
#
python run_ner_crf.py \
  --model_type=bert \
  --train_data_path ../data/test_data/train.txt \
  --dev_data_path ../data/test_data/dev.txt \
  --label_txt ../data/label_dir/label_crf.txt \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --markup "bio" \
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