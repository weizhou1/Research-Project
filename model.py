# !/usr/bin/env python
# -*- coding: utf8 -*-



from pickle import FALSE
import sys
from matplotlib.cbook import flatten
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import glorot_init
from modules import GCN, AvgReadout, Discriminator
from collections import Counter
from evaluation import *
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")
import gc
from collections import Counter
from evaluation import validate_performance,validate_ARI_NMI
import time
from utils import sparse_to_tuple
import random
import itertools

from datetime import datetime

# from numpy.linalg import norm
# from scipy.spatial.distance import cityblock
import networkx as nx
from networkx.algorithms import bipartite


def Graph_Diffusion_Convolution(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A #Self-loops
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return sp.csr_matrix(T_S)


def get_past_emb(node,emb): # get the embeding of this bin 
    new_past_emb = np.zeros(emb[0].shape)
    count = 0
    for e in node:
        # print("e past node length",len(past_top_nodes[i]))
        new_past_emb += emb[e]
        count+=1
    new_past_emb /= count
    return new_past_emb


def satisfy_neg_con(past_tn,bn,pairs): # check if given tuple violate given negative constarin 
    if any(item in past_tn for item in pairs[bn]):
        return False

    return True

def add_match_in_pastnode(past_tn,match): # update new node in each bin  & Update the embed of node in each bin
    for k, v in match.items():
        past_tn[int(v)].append(k)

    return past_tn

def matching(constraints,embeds_np,samples,valid_con,ground_truth):
    # # order the constrain 
    # l_count = [0]* len(constraints)# count of the constrain node occurance 
    # flatten_con = np.hstack(constraints).tolist()
    # for i in range(len(constraints)):
    #     for node in constraints[i]:
    #         l_count[i] +=flatten_con.count(node)
    # idx = np.argsort(np.array(l_count))[::-1]
    # constraints[:] = [constraints[i] for i in idx] # sort list by the occurence of node
    # seen = set()
    # constraints = [x for x in constraints if frozenset(x) not in seen and not seen.add(frozenset(x))] # get rid of duplicate 

    pairs_dict = {} # convert the constrain to dictionary to improve the efficiency (much quicker than before)
    for k, v in samples.tolist():
        pairs_dict[k] = pairs_dict.get(k, ()) + (v,)
    
    print("----  Finish order constarin -----")

    B = nx.Graph()
    N = max([len(x) for x in constraints])
    for n in constraints: # initiate the bins by adding n nodes in it
        if len(n) == N:
            top_nodes = n
            constraints.remove(n)
            break
        # if N <= 19: # for CAMI
        #     if len(n) == N:
        #         top_nodes = n
        #         break

    past_top_nodes =  [[i]for i in top_nodes]

    n = 0
    # match_n = 2
    print("length of constrains",len(constraints))
    while n < len(constraints):
        bottom_nodes = constraints[n]  # keep update bottom nodes
        for tn in [item for sublist in past_top_nodes for item in sublist]: # if bottom node not already in the bin
            if tn in bottom_nodes:
                bottom_nodes.remove(tn)
        if len(bottom_nodes) > 0:
            B.add_nodes_from([str(x) for x in top_nodes], bipartite=0)
            B.add_nodes_from(bottom_nodes, bipartite=1)
            # add edges of connected nodes 
            added_edge = False
            cur_bottom_node =[]
            cur_top_node =[]
            # for i in range(N): # interate through all bins
            # for i in range(n_clusters): # interate through all bins
            for i in range(len(past_top_nodes)):
                weight = {}
                for j in range(len(bottom_nodes)): 
                    if bottom_nodes[j] in valid_con: # confirm that node exist in the ground truth
                        if satisfy_neg_con(past_top_nodes[i],bottom_nodes[j],pairs_dict):
                            new_past_emb = get_past_emb(past_top_nodes[i], embeds_np)
                            w = np.linalg.norm(new_past_emb-np.array(embeds_np[bottom_nodes[j]]))
                            # w = cityblock(new_past_emb,np.array(embeds_np[bottom_nodes[j]]))
                            B.add_edge(str(i),bottom_nodes[j],weight = w)
                            cur_bottom_node.append(bottom_nodes[j])
                            cur_top_node.append(str(i))
                            added_edge = True
    
            if added_edge:
                #Obtain the minimum weight full matching i.e find the minimum distance and satisfy the negative constrain
                cur_bottom_node = list(set(cur_bottom_node))
                cur_top_node = list(set(cur_top_node))

                if len(cur_bottom_node) <= len(cur_top_node):
                    try:
                        my_matching = bipartite.matching.minimum_weight_full_matching(B, cur_bottom_node,"weight")	
                    except ValueError:
                        print("ValueError: cost matrix is infeasible")
                    else:
                        idx = int(len(my_matching)/2)
                        my_matching = dict(list(my_matching.items())[:idx])
                        past_top_nodes = add_match_in_pastnode(past_top_nodes,my_matching)

                else:
                    try:
                        my_matching = bipartite.matching.minimum_weight_full_matching(B,cur_top_node,"weight")	
                    except ValueError:
                        print("ValueError: cost matrix is infeasible")
                    else:
                        idx = int(len(my_matching)/2)
                        my_matching = dict(list(my_matching.items())[idx:])
                        past_top_nodes = add_match_in_pastnode(past_top_nodes,my_matching)

        B = nx.Graph()
        n +=1
    
    return past_top_nodes


class Learning:
    def __init__(self, ipt_dim, hid_dim, opt_dim, args):
        self.args = args
        self.model = RepBin(ipt_dim, hid_dim, opt_dim, 'prelu')
        self.model = self.model.to(device)

    def train(self, adj, diff,feats, Gx, samples, constraints, ground_truth,assembly_graph,composition):
        n_nodes = adj[2][0]
        adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].T),torch.FloatTensor(adj[1]),torch.Size(adj[2])).to(device)
        diff = torch.sparse.FloatTensor(torch.LongTensor(diff[0].T),torch.FloatTensor(diff[1]),torch.Size(diff[2])).to(device)
        feats = torch.FloatTensor(feats[np.newaxis]).to(device)
        samples = torch.LongTensor(samples).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        b_xent = nn.BCEWithLogitsLoss()
        cnt_wait, best, best_t = 0, 1e9, 0


        print("### Step 1: Constraint-based Learning model.")
        list_loss,list_losss,list_lossc = [],[],[]
        list_p,list_r,list_f1,list_ari = [],[],[],[]
        for epoch in range(self.args.epochs):
            self.model.train()
            optimizer.zero_grad()
            # corruption
            rnd_idx = np.random.permutation(n_nodes)
            shuf_fts = feats[:,rnd_idx,:].to(device)   
            
            # labels
            lbl_1 = torch.ones(self.args.batch_size, n_nodes)
            lbl_2 = torch.zeros(self.args.batch_size, n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

            logits, hidds = self.model(feats, shuf_fts, adj, diff,True, None, None)
            loss_s = b_xent(logits, lbl)
            # loss_c = self.model.constraints_loss(hidds, samples)
            # loss = self.args.lamb*loss_s + (1-self.args.lamb)*loss_c
            loss = loss_s
             

            if epoch+1 == 1 or (epoch+1)%100 == 0:
                # print("Epoch: {:d} loss={:.5f} loss_s={:.5f} loss_c={:.5f}".format(epoch+1,loss.item(),loss_s.item(),loss_c.item()))
                print("Epoch: {:d} loss={:.5f} loss_s={:.5f} ".format(epoch+1,loss.item(),loss_s.item()))

            if loss < best:
                cnt_wait = 0
                best,best_t = loss,epoch
                torch.save(self.model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait+=1
            if cnt_wait == self.args.patience:
                print('Early stopping!')
                break

            loss.backward()
            optimizer.step()
        print('Loading {}-th epoch.'.format(best_t+1))
        self.model.load_state_dict(torch.load('best_model.pkl'))
        self.model.eval()
        embeds, _ = self.model.embed(feats, adj,diff, True)
        print("### Optimization Finished!")
        true_labels = ground_truth

        lbls_idx = [k for k,v in true_labels.items()]
        cons = [val for line in constraints for val in line if val in lbls_idx]
        valid_con = cons # cons that appear in the ground truth
        cons = [k for k,v in Counter(cons).items() if v>3]

        
        n_clusters = len(Counter([true_labels[c] for c in cons]))
        embs = embeds.cpu().detach().numpy()[cons]
        labels = list(set([true_labels[i] for i in cons]))
        labels_map = {idx:i for i,idx in enumerate(labels)}
        lbls = {i:true_labels[idx] for i,idx in enumerate(cons)}

        # # MixBin-Learning test
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=self.args.n_clusters)
        # y_pred = kmeans.fit_predict(embeds.cpu().detach().numpy())
        # pred_labels = {i:j for i,j in enumerate(y_pred)}
        # # p, r, ari, f1 = validate_performance(lbls, pred_labels)
        # cons = range(0,len(embeds))
        # init_labels_dict = {cons[i]:y_pred[i] for i in range(len(y_pred))}
        # sys.exit()


        match = matching(constraints,embeds.cpu().detach().numpy(),samples,valid_con,ground_truth)
        init_labels_dict = dict()
        for i in range(len(match)):
            tmp = {}
            tmp = tmp.fromkeys(match[i], i)
            init_labels_dict.update(tmp)

        ### GCN-Label Annotation
        print()
        print("### Step 2: Constraint-based Binning model.")
        idxs = [idx for idx,val in init_labels_dict.items()]
        mask = np.array([True if idx in idxs else False for idx in range(n_nodes)])
        init_labels = [init_labels_dict[idx] if idx in idxs else 0 for idx in range(n_nodes)]
        init_labels = torch.LongTensor(init_labels).to(device)
        mask = torch.LongTensor(mask).to(device)
        listg_loss,loss_last = [],1e9
        for epoch in range(1000):
            self.model.train()
            optimizer.zero_grad()
            out = self.model.labelProp(feats, adj,diff,True)
            loss = F.cross_entropy(out, init_labels, reduction='none')
            mask = mask.float()
            mask = mask / mask.mean()
            loss *= mask
            loss = loss.mean()
            listg_loss.append(loss.item())
            # loss += self.args.weight_decay * self.model.l2_loss()

            pred = out.argmax(dim=1)
            pred_dict = {i:j.item() for i,j in enumerate(pred)}
            p, r, ari, f1 = validate_performance(ground_truth, pred_dict)

            if loss_last-loss < 0.001:
                print('Early stopping!')
                break
            else:
                loss_last = loss
                torch.save(self.model.state_dict(), 'best_model_lp.pkl')

            if epoch+1 == 1 or (epoch+1)%10 == 0:
                print("Epoch: {:d} loss={:.5f}".format(epoch+1,loss.item()))

            loss.backward()
            optimizer.step()
        
        self.model.load_state_dict(torch.load('best_model_lp.pkl'))
        self.model.eval()
        out = self.model.labelProp(feats, adj, diff,True)
        pred = out.argmax(dim=1)
        pred_dict = {i:j.item() for i,j in enumerate(pred)}
        # print("predict doct",pred_dict)
        return pred_dict


class RepBin(nn.Module):
    def __init__(self, n_in, n_h, n_opt, act):
        super(RepBin, self).__init__()
        self.gcn = GCN(n_in, n_h, act)
        self.readout = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.gcn2 = GCN(n_h, n_opt, 'prelu')

    # def forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2):
    #     h_1 = self.gcn(seq1, adj, sparse)
    #     c = self.readout(h_1)
    #     c = self.sigm(c)
    #     h_2 = self.gcn(seq2, adj, sparse)
    #     ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
    #     return ret, h_1.squeeze(0)
        
        
    def forward(self, seq1, seq2, adj, diff, sparse, samp_bias1, samp_bias2):
        # h_1 = self.gcn(seq1, adj, sparse)
        h_1 = self.gcn(seq1, diff, sparse)
        c = self.readout(h_1)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, diff, sparse)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret, h_1.squeeze(0)

    def embed(self, seq, adj, diff,sparse):
        # h_1 = self.gcn(seq, adj, sparse)
        h_1 = self.gcn(seq, diff, sparse)
        c = self.readout(h_1)
        h_1 = h_1.squeeze(0)
        # return h_1.detach().numpy(), c.detach()
        return h_1, c

    def labelProp(self, seq, adj, diff,sparse):
        h = self.gcn(seq, diff, sparse)
        h = self.gcn2(h, diff, sparse)
        # h = F.log_softmax(self.gcn2(h, adj, sparse))
        return h.squeeze(0)
        # return h

    # def l2_loss(self):
    #     loss = None
    #     for p in self.gcn2.parameters():
    #         if loss is None:
    #             loss = p.pow(2).sum()
    #         else:
    #             loss += p.pow(2).sum()
    #     return loss

    # def constraints_loss(self, embeds, constraints):
    #     neg_pairs = torch.stack([constraints[:, 0], constraints[:, 1]], 1)
    #     p = torch.index_select(embeds, 0, neg_pairs[:,0])
    #     q = torch.index_select(embeds, 0, neg_pairs[:,1])
    #     return torch.exp(-F.pairwise_distance(p, q, p=2)).mean()

