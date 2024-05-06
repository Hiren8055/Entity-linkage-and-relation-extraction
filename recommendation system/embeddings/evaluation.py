
import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math

from itertools import groupby

from utils import Triple, getRel

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics.pairwise import pairwise_distances

# from projection import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

def isHit10(triple, tree, cal_embedding, tripleDict, isTail):
	# If isTail == True, evaluate the prediction of tail entity
	if isTail == True:
		k = 0
		wrongCount = 0
		while wrongCount < 10:
			k += 15
			tail_dist, tail_ind = tree.query(cal_embedding, k=k)
			for elem in tail_ind[0][k - 15: k]:
				if triple.t == elem:
					return True
				elif (triple.h, elem, triple.r) in tripleDict:
					continue
				else:
					wrongCount += 1
					if wrongCount > 9:
						return False
	# If isTail == False, evaluate the prediction of head entity
	else:
		k = 0
		wrongCount = 0
		while wrongCount < 10:
			k += 15
			head_dist, head_ind = tree.query(cal_embedding, k=k)
			for elem in head_ind[0][k - 15: k]:
				if triple.h == elem:
					return True
				elif (elem, triple.t, triple.r) in tripleDict:
					continue
				else:
					wrongCount += 1
					if wrongCount > 9:
						return False

# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == tail:
			return wrongAnswer
		elif (head, num, rel) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == head:
			return wrongAnswer
		elif (num, tail, rel) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

def pairwise_L1_distances(A, B):
	dist = torch.sum(torch.abs(A.unsqueeze(1) - B.unsqueeze(0)), dim=2)
	return dist

def pairwise_L2_distances(A, B):
	AA = torch.sum(A ** 2, dim=1).unsqueeze(1)
	BB = torch.sum(B ** 2, dim=1).unsqueeze(0)
	dist = torch.mm(A, torch.transpose(B, 0, 1))
	dist *= -2
	dist += AA
	dist += BB
	return dist


def projection_transR_pytorch(original, proj_matrix):
	ent_embedding_size = original.shape[1]
	rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
	original = original.view(-1, ent_embedding_size, 1)
	proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
	return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)



def evaluation_transR_helper(testList, tripleDict, ent_embeddings, 
	rel_embeddings, proj_embeddings, L1_flag, filter, head):
	# embeddings are torch tensor like (No Variable!)
	# Only one kind of relation

	headList = [triple.h for triple in testList]
	tailList = [triple.t for triple in testList]
	relList = [triple.r for triple in testList]

	h_e = ent_embeddings[headList]
	t_e = ent_embeddings[tailList]
	r_e = rel_embeddings[relList]
	this_rel = relList[0]
	this_proj_emb = proj_embeddings[[this_rel]]
	this_proj_all_e = projection_transR_pytorch(ent_embeddings, this_proj_emb)
	this_proj_all_e = this_proj_all_e.cpu().numpy()

	if head == 1:
		proj_t_e = projection_transR_pytorch(t_e, this_proj_emb)
		c_h_e = proj_t_e - r_e
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListHead)
		hit10Count = len(isHit10ListHead)
		tripleCount = len(rankListHead)

	elif head == 2:
		proj_h_e = projection_transR_pytorch(h_e, this_proj_emb)
		c_t_e = proj_h_e + r_e
		c_t_e = c_t_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		totalRank = sum(rankListTail)
		hit10Count = len(isHit10ListTail)
		tripleCount = len(rankListTail)

	else:
		proj_h_e = projection_transR_pytorch(h_e, this_proj_emb)
		c_t_e = proj_h_e + r_e
		proj_t_e = projection_transR_pytorch(t_e, this_proj_emb)
		c_h_e = proj_t_e - r_e

		c_t_e = c_t_e.cpu().numpy()
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListTail) + sum(rankListHead)
		hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
		tripleCount = len(rankListTail) + len(rankListHead)

	return hit10Count, totalRank, tripleCount

class MyProcessTransR(multiprocessing.Process):
	def __init__(self, L, tripleDict, ent_embeddings, 
		rel_embeddings, proj_embeddings, L1_flag, filter, queue=None, head=0):
		super(MyProcessTransR, self).__init__()
		self.L = L
		self.queue = queue
		self.tripleDict = tripleDict
		self.ent_embeddings = ent_embeddings
		self.rel_embeddings = rel_embeddings
		self.proj_embeddings = proj_embeddings
		self.L1_flag = L1_flag
		self.filter = filter
		self.head = head

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
			except:
				time.sleep(5)
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
			self.queue.task_done()

	def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
		proj_embeddings, L1_flag, filter, L, head):

		hit10Count, totalRank, tripleCount = evaluation_transR_helper(testList, tripleDict, ent_embeddings, 
			rel_embeddings, proj_embeddings, L1_flag, filter, head=head)

		L.append((hit10Count, totalRank, tripleCount))

def evaluation_transR(testList, tripleDict, ent_embeddings, rel_embeddings, 
	proj_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
	# embeddings are torch tensor like (No Variable!)

	if k > len(testList):
		testList = random.choices(testList, k=k)
	elif k > 0:
		testList = random.sample(testList, k=k)

	# Split the testList according to the relation
	testList.sort(key=lambda x: (x.r, x.h, x.t))
	grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

	ent_embeddings = ent_embeddings.cpu()
	rel_embeddings = rel_embeddings.cpu()
	proj_embeddings = proj_embeddings.cpu()

	with multiprocessing.Manager() as manager:
		L = manager.list()
		queue = multiprocessing.JoinableQueue()
		workerList = []
		for i in range(num_processes):
			worker = MyProcessTransR(L, tripleDict, ent_embeddings, rel_embeddings,
				proj_embeddings, L1_flag, filter, queue=queue, head=head)
			workerList.append(worker)
			worker.daemon = True
			worker.start()

		for k, subList in grouped:
			queue.put(subList)

		queue.join()

		resultList = list(L)

		for worker in workerList:
			worker.terminate()

	if head == 1 or head == 2:
		hit10 = sum([elem[0] for elem in resultList]) / len(testList)
		meanrank = sum([elem[1] for elem in resultList]) / len(testList)	
	else:
		hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
		meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

	print('Meanrank: %.6f' % meanrank)
	print('Hit@10: %.6f' % hit10)

	return hit10, meanrank

