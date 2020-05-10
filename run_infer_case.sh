export CUDA_VISIBLE_DEVICES=3
data_dir=/path/to/your/data
python run_infer.py --mode fine-tune --data_dir ${data_dir}/MRPC/ --data_sign MRPC
python run_infer.py --mode score --data_dir ${data_dir}/MRPC/ --data_sign MRPC
python run_infer.py --mode attack --adv_type greedy --data_dir ${data_dir}/MRPC/ --data_sign MRPC
python run_infer.py --mode attack --adv_type random  --random_attack_file ${data_dir}/MRPC/mrpc.tsv --data_dir ${data_dir}/MRPC/ --data_sign MRPC
