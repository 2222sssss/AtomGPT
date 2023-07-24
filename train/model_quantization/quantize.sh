export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export CUDA_HOME=/usr/local/cuda-11.8/
CUDA_VISIBLE_DEVICES=1,2,3 python quantize_model.py \
    --model_folder /mnt/data/zhangzheng/data/atomgpt/model_sft_atomgpt_56000_0717/checkpoint-11500_merge \
    --example_file_path /mnt/data/zhangzheng/data/dataset/指令微调/atom_from_chatgpt/wanjuan/qa_task_train/train_atomgpt_qa.csv \
                        /mnt/data/zhangzheng/data/dataset/指令微调/wanjuan_data/qa_task_answer_long/train_select_standard_sharegpt_90k.csv \
                        /mnt/data/zhangzheng/data/dataset/指令微调/wanjuan_data/qa_task_answer_long/train_select_standard_atomecho_3k.csv \
    --group_size 128 \
    --bits 4 \
    --tokenizer_fast false \
    --quant_batch_size 4 \
    --samples_num 3000 \
    --use_triton false 