
import os
import math
import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from evaluation import projection_transR_pytorch

# from projection import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor


# TransR without using pretrained embeddings,
# i.e, the whole model is trained from scratch.
class TransRModel(nn.Module):
	def __init__(self, config):
		super(TransRModel, self).__init__()
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.ent_embedding_size = config.ent_embedding_size
		self.rel_embedding_size = config.rel_embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.batch_size = config.batch_size

		ent_weight = torch.FloatTensor(self.entity_total, self.ent_embedding_size)
		rel_weight = torch.FloatTensor(self.relation_total, self.rel_embedding_size)
		proj_weight = torch.FloatTensor(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
		# initializing the weights with random value from uniform distribution range between -1 to 1 #########
		nn.init.xavier_uniform_(ent_weight) 
		nn.init.xavier_uniform_(rel_weight)
		nn.init.xavier_uniform_(proj_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.ent_embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size)
		self.proj_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)
		self.proj_embeddings.weight = nn.Parameter(proj_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		# normalize_proj_emb = F.normalize(self.proj_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb
		# self.proj_embeddings.weight.data = normalize_proj_emb

	def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)
		pos_proj = self.proj_embeddings(pos_r)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)
		neg_proj = self.proj_embeddings(neg_r)

		pos_h_e = projection_transR_pytorch(pos_h_e, pos_proj)
		pos_t_e = projection_transR_pytorch(pos_t_e, pos_proj)
		neg_h_e = projection_transR_pytorch(neg_h_e, neg_proj)
		neg_t_e = projection_transR_pytorch(neg_t_e, neg_proj)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
		else:
			pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
		return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e	

# TransR with using pretrained embeddings.
# Pretrained embeddings are trained with TransE, stored in './transE_%s_%s_best.pkl',
# with first '%s' dataset name,
# second '%s' embedding size.
# Initialize projection matrix with identity matrix.
class TransRPretrainModel(nn.Module):
	def __init__(self, config):
		super(TransRPretrainModel, self).__init__()
		self.dataset = config.dataset
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.ent_embedding_size = config.ent_embedding_size
		self.rel_embedding_size = config.rel_embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.batch_size = config.batch_size
		
		# with open('model\FB13\l_0.001_es_0_L_1_em_100_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransE.ckpt', 'rb') as fr:
		with open('./transE_%s_%s_best.pkl' % (config.dataset, str(config.ent_embedding_size)), 'rb') as fr:
		# model = 
		# model.load_state_dict(torch.load('model\FB13\l_0.001_es_0_L_1_em_100_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransE.ckpt'))
			
			ent_embeddings_list = pickle.load(fr)
			rel_embeddings_list = pickle.load(fr)

		ent_weight = floatTensor(ent_embeddings_list)
		rel_weight = floatTensor(rel_embeddings_list)
		proj_weight = floatTensor(self.rel_embedding_size, self.ent_embedding_size)
		nn.init.eye(proj_weight)
		proj_weight = proj_weight.view(-1).expand(self.relation_total, -1)

		self.ent_embeddings = nn.Embedding(self.entity_total, self.ent_embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size)
		self.proj_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)
		self.proj_embeddings.weight = nn.Parameter(proj_weight)

	def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)
		pos_proj = self.proj_embeddings(pos_r)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)
		neg_proj = self.proj_embeddings(neg_r)

		pos_h_e = projection_transR_pytorch(pos_h_e, pos_proj)
		pos_t_e = projection_transR_pytorch(pos_t_e, pos_proj)
		neg_h_e = projection_transR_pytorch(neg_h_e, neg_proj)
		neg_t_e = projection_transR_pytorch(neg_t_e, neg_proj)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
		else:
			pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
		return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e
