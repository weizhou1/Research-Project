# !/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import argparse
import networkx as nx
import time
from loader import *
from utils import sparse_to_tuple
from utils import normalize_adj
from model import Learning
from model import Graph_Diffusion_Convolution
from evaluation import *
from utils import sample_constraints
import warnings
from cov_comp_loader import load_feat
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
import time

parser = argparse.ArgumentParser('Unsup learning model.')
parser.add_argument('--dataset', type=str, default='Sim-5G', help='Dataset string.')
# parser.add_argument('--n_clusters', type=int, default=100,help='Number of clusters.')
parser.add_argument('--alpha', type=float, default=0.01, help='Teleport probability in graph diffusion convolution operators.')
parser.add_argument('--eps', type=float, default=0.0001, help='Threshold epsilon to sparsify in graph diffusion convolution operators.')

parser.add_argument('--lr', type=float, default=0.005, help='Number of learning rate.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs.')
parser.add_argument('--hid_dim', type=int, default=32, help='Dimension of hidden2.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping (# of epochs).')
parser.add_argument('--lamb', type=float, default=0.2, help='Weight to balance constraints.')


# Composition argument 
parser.add_argument('-f','--fasta_file', default = "Sim-5G",help='specify the fasta file.')
parser.add_argument('-k','--kmer_length', type=int, default=4, help='specify kmer length, default 4.')
parser.add_argument('-l','--threshold_length', type=int, default=0, help='specify the sequence length threshold, contigs shorter than this value will not be included. Defaults to 1000.')

args = parser.parse_args()


print('----------args----------')
for k in list(vars(args).keys()):
	print('%s: %s' % (k, vars(args)[k]))
print('----------args----------\n')


def main():

	assembly_graph, constraints, ground_truth, Gx ,filtered_nodes = load_data(args.dataset)
	triplets = sample_constraints(constraints, ground_truth)

	diff = Graph_Diffusion_Convolution(assembly_graph, args.alpha, args.eps)
	diff = sparse_to_tuple(diff)
	
	adj = normalize_adj(assembly_graph + sp.eye(assembly_graph.shape[0]))
	adj = sparse_to_tuple(adj)

	feats = assembly_graph.todense() 

		
	composition = np.array(load_feat(args))
	composition = np.delete(composition,filtered_nodes, axis= 0)
	composition = normalize(composition, axis=1, norm='l1')
	feats = np.concatenate((composition, feats), axis=1) 
	

	# Prepocess constraints
	# first order the constrain  
	l_count = [0]* len(constraints)# count of the constrain node occurance 
	flatten_con = np.hstack(constraints).tolist()
	for i in range(len(constraints)):
		for node in constraints[i]:
			l_count[i] +=flatten_con.count(node)
	idx = np.argsort(np.array(l_count))[::-1]
	constraints[:] = [constraints[i] for i in idx] # sort list by the occurence of node
	seen = set()
	constraints = [x for x in constraints if frozenset(x) not in seen and not seen.add(frozenset(x))] # get rid of duplicate 
	n_clusters = max([len(x) for x in constraints])
	print("bin size = ", n_clusters)


	model = Learning(feats.shape[1], args.hid_dim,n_clusters, args)
	pred_labels = model.train(adj, diff,feats, Gx, triplets, constraints, ground_truth, assembly_graph,composition)
	
	print("\nEvaluation:")
	p,r,f1,ari,nmi = evaluate_performance(ground_truth, pred_labels)
	print ("### Precision = %0.4f, Recall = %0.4f, F1 = %0.4f, ARI = %0.4f, NMI = %0.4f"  % (p,r,f1,ari,nmi))


	# np.save('plot/Yujia/data/Preds_MixBin_'+ args.dataset+'.npy',pred_labels)

	return p,r,f1,ari,nmi 


if __name__ == '__main__':

	main()
