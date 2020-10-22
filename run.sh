#Serial fb15k237
python -u acre.py --data FB15k-237 --batch 128 \
--hid_drop 0.5 --feat_drop 0.2 --lr 0.001 --inp_drop 0.3  --way s --train_strategy one_to_x

#Serial fb15k
python -u acre.py --data FB15k --batch 256 \
--hid_drop 0.2 --feat_drop 0.2 --lr 0.001 --inp_drop 0.2  --way s  --train_strategy one_to_n

#Serial kinship
python -u acre.py --data kinship --batch 128 \
--hid_drop 0.5 --feat_drop 0.5 --lr 0.001 --inp_drop 0.2  --way s --train_strategy one_to_n

#Serial WN18RR
python -u acre.py --data WN18RR --batch 256 \
--hid_drop 0.5 --feat_drop 0.1 --lr 0.00125 --inp_drop 0.2  --way s --train_strategy one_to_n

#Serial WN18
python -u acre.py --data WN18 --batch 256 \
--hid_drop 0.3 --feat_drop 0.3 --lr 0.0012 --inp_drop 0.2  --way s  --train_strategy one_to_n

#Serial DB100K
python -u acre.py --data DB100K --batch 256 \
--hid_drop 0.3 --feat_drop 0.2 --lr 0.0012 --inp_drop 0.2  --way s --train_strategy one_to_x

#Parallel fb15k237
python -u acre.py --data FB15k-237 --batch 128 \
--hid_drop 0.5 --feat_drop 0.2 --lr 0.001 --inp_drop 0.3  --way p --train_strategy one_to_x

#Parallel fb15k
python -u acre.py --data FB15k --batch 256 \
--hid_drop 0.2 --feat_drop 0.2 --lr 0.001 --inp_drop 0.2  --way p  --train_strategy one_to_n

#Parallel kinship
python -u acre.py --data kinship --batch 128 \
--hid_drop 0.5 --feat_drop 0.2 --lr 0.0001 --inp_drop 0.3  --way p  --train_strategy one_to_n

#Parallel WN18RR
python -u acre.py --data WN18RR --batch 256 \
--hid_drop 0.5 --feat_drop 0.1 --lr 0.00125 --inp_drop 0.3  --way p --train_strategy one_to_x

#Parallel WN18
python -u acre.py --data WN18 --batch 256 \
--hid_drop 0.3 --feat_drop 0.3 --lr 0.0012 --inp_drop 0.2  --way p --train_strategy one_to_x

#Parallel DB100K
python -u acre.py --data DB100K --batch 256 \
--hid_drop 0.3 --feat_drop 0.2 --lr 0.0012 --inp_drop 0.2  --way p --train_strategy one_to_x