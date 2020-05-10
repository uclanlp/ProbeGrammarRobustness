export CUDA_VISIBLE_DEVICES=3
layer=1
for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12;do
    python trans_sent_acc.py --data_file prep2.txt  --type prep --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file art2.txt  --type art --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file vt2.txt  --type vt --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file sva2.txt  --type sva --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file wform2.txt  --type wform --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file worder2.txt  --type worder --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file nn2.txt  --type nn --layer ${layer} --transformer-model roberta-base
    python trans_sent_acc.py --data_file tran2.txt  --type tran --layer ${layer} --transformer-model roberta-base
done

