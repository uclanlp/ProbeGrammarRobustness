export CUDA_VISIBLE_DEVICES=3
offset=1
for offset in $(seq -3 -1) $(seq 1 3);do
    python -u bert_mlm_dis.py --type SVA --offset ${offset}
done

