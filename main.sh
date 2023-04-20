dataroot=/home/tuantran/CAGrad/multi_mnist/data/ # folder root data
weight=equal
dataname=multi_fashion_and_mnist # multi_mnist / multi_fashion / multi_fashion_and_mnist
method=cagrad # mgd / cagrad / pcgrad / adacagrad
alpha=0.2
bs=256
seed=0
python3 -u main.py --dataname $dataname --bs $bs --dataroot $dataroot --seed $seed --weight $weight --method $method --alpha $alpha 