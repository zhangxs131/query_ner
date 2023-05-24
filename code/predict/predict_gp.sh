python predict_gp.py \
  --label_list_txt ../data/label_dir/label.txt   \
  --test_data   ../data/test_data/test.txt  \
  --best_model_name ../outputs/gp/queryner_output/ \
  --result_name ../results/gp/result_0406.csv \
  --batch_size 32 \
  --device 0 \
  --seed 2023