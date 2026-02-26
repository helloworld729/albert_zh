python create_pretraining_data.py \
    --input_file=../data/news_zh_1.txt \
    --output_file=output.json \
    --vocab_file=../albert_config/vocab.txt \
    --do_lower_case \
    --do_whole_word_mask \
    --max_seq_length=521 \
    --max_predictions_per_seq=51 \
    --dupe_factor=10