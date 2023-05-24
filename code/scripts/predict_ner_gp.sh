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
