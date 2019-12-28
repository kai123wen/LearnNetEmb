/root/anaconda3/envs/ne/bin/python3.6 -u

 python main_AIDW.py --input_edgelist input/citeseer-edgelist.txt --input_ppmi input/citeseer-PPMI-4.mat --input_adj input/citeseer-undirected.mat --walk_length 80 --num_walks 10 --window_size 10 --batch_size 50 --K 5 --hidden_layers 1 --hidden_neurons 128 --rep 'output/citeseer-rep.mat' --resultTxt 'results/AIDW-citeseer.txt'

