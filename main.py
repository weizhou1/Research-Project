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

# args = parser.parse_args()

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
	# assembly_graph, constraints, ground_truth, Gx  = load_data(args.dataset)
	triplets = sample_constraints(constraints, ground_truth)

	diff = Graph_Diffusion_Convolution(assembly_graph, args.alpha, args.eps)
	diff = sparse_to_tuple(diff)
	
	adj = normalize_adj(assembly_graph + sp.eye(assembly_graph.shape[0]))
	adj = sparse_to_tuple(adj)

	feats = assembly_graph.todense() 
	# print("assembly_graph",feats.shape)
	# print("edge num :",np.matrix.sum(feats))

		
	# composition, coverage = load_feat(args)
	composition = np.array(load_feat(args))
	# composition = np.array(composition) 
	# coverage = np.array(coverage) 
	composition = np.delete(composition,filtered_nodes, axis= 0)
	# print(composition)
	# coverage = np.delete(coverage,filtered_nodes, axis= 0)

	# normed_coverage= normalize(coverage, axis = 0 ,norm='l1')
	# normed_composition= normalize(composition, axis = 1, norm='l1')

	# print("coverage",coverage.shape)
	# print(coverage)
	
	# normed_comp_cov = composition * coverage[:, None]                     
	# normed_comp_cov = normalize(normed_comp_cov, axis=1, norm='l1')
	# print("normalise coposiiton and coverage ",normed_comp_cov.shape)
	composition = normalize(composition, axis=1, norm='l1')
	# print("normalise coposiiton ",composition.shape)
	# print(composition)
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



	# model = Learning(feats.shape[1], args.hid_dim, args.n_clusters, args)
	model = Learning(feats.shape[1], args.hid_dim,n_clusters, args)
	pred_labels = model.train(adj, diff,feats, Gx, triplets, constraints, ground_truth, assembly_graph,composition)
	
	print("\nEvaluation:")
	p,r,f1,ari,nmi = evaluate_performance(ground_truth, pred_labels)
	print ("### Precision = %0.4f, Recall = %0.4f, F1 = %0.4f, ARI = %0.4f, NMI = %0.4f"  % (p,r,f1,ari,nmi))


	np.save('plot/Yujia/data/Preds_MixBin_'+ args.dataset+'.npy',pred_labels)

	return p,r,f1,ari,nmi 


if __name__ == '__main__':
	# # get the start time
	# st = time.process_time()

	main()

	# # get the end time
	# et = time.process_time()

	# # get execution time
	# res = et - st
	# print('CPU Execution time:', res/60, 'minutes')
	

	# N = 5
	# perfom = [[] for _ in range(N)]
	# cpu_time = [[] for _ in range(N)]
	# for i in range(5):
	# 	# get start time
	# 	st = time.process_time()
	# 	p,r,f1,ari,nmi = main()
	# 	# get the end time
	# 	et = time.process_time()
	# 	perfom[i] = [p,r,f1,ari,nmi]

	# 	# get execution time
	# 	res = et - st
	# 	print('CPU Execution time:', res/60, 'minutes')
	# 	cpu_time[i] = res

	# print("performance")
	# for i in range(N):
	# 	print(perfom[i])
	# 	print('CPU Execution time:',cpu_time[i],"seconds")
	# 	print("\n")

