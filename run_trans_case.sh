export CUDA_VISIBLE_DEVICES=3
data_dir=/path/to/your/own/data

#bert
python run_transformers.py --mode fine-tune --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC 
python run_transformers.py --mode score --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC
python run_transformers.py --mode attack --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC --adv_type greedy
python run_transformers.py --mode attack --target_model bert --model_name_or_path bert-base-uncased --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC --adv_type random --random_attack_file ${data_dir}/MRPC/mrpc.tsv

#roberta
python run_transformers.py --mode fine-tune --target_model roberta --model_name_or_path roberta-base --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC 
python run_transformers.py --mode score --target_model roberta --model_name_or_path roberta-base --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC
python run_transformers.py --mode attack --target_model roberta --model_name_or_path roberta-base --do_lower_case --data_dir ${data_dir}/MRPC --data_sign MRPC --adv_type greedy
