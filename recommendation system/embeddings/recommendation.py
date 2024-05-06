import argparse
from sentence_transformers import SentenceTransformer
sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import nltk
from nltk.corpus import stopwords
import re
from neo4j import GraphDatabase
import neo4j
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import pickle

import sys

sys.path.append("C:\Storage\Major project\Entity-linkage-and-relation-extraction\recommendation system\embeddings")
# from model import *
from model import TransRModel
# import __main__
# setattr(__main__, "TransRModel", TransRModel)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# with open(r"C:\Storage\Major project\Entity-linkage-and-relation-extraction\recommendation system\embeddings\model\newscout_1\l1_0.001_l2_0.0005_es_0_L_1_eem_100_rem_100_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransR.ckpt", 'rb') as fr:
# 		# model = 
# 		# model.load_state_dict(torch.load('model\FB13\l_0.001_es_0_L_1_em_100_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransE.ckpt'))
			
# 			ent_embeddings_list = pickle.load(fr)
# 			rel_embeddings_list = pickle.load(fr)

# sent_model = torch.load(r"C:\Storage\Major project\Entity-linkage-and-relation-extraction\recommendation system\embeddings\model\newscout_1\l1_0.001_l2_0.0005_es_0_L_1_eem_100_rem_100_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransR.ckpt")
# print(ent_embeddings_list)
# print(sent_model)

parser = argparse.ArgumentParser(description='recommendation system')
parser.add_argument("id")
parser.add_argument("training")

driver = GraphDatabase.driver(uri="bolt://localhost:7687",auth =("neo4j","12345678"))
session =driver.session()

def feature_embedding(sent, sent_model):
    # print("sent", sent)

    text = re.sub(r'\[[0-9]*\]',' ',sent)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    sentence = nltk.sent_tokenize(text)
    # print()
    sentence =  [nltk.word_tokenize(sent) for sent in sentence]
    for i in range(len(sentence)):
        sentence[i] = [word for word in sentence[i] if word not in stopwords.words('english')]
    

    # print("sentence",sentence)
    # sentence = torch.FloatTensor(sentence[0],100)
    # embedding = sent_model.ent_embeddings(sentence)
    # print(embedding[0])
    # print(len(sentence))
    if len(sentence) > 1: 
        # print(" 1")

        # if len(sentence) >= 2:
        #     print(" 2") 
        temp = [] 
        
        for i in range(len(sentence)):
            for j in range(len(sentence[i])):
                print(sentence[i][j])
                temp.append(sentence[i][j])
        # print("temp",temp)
        # embedding = sent_model.ent_embeddings(temp)
        # print(embedding)
        embedding = sent_model.encode(temp)
        return embedding.tolist()
        
        # embedding = sent_model.encode(sentence)
        # return embedding.tolist()

    

    elif len(sentence) == 1:
        # print("equal 1")
        # print(type(sentence))
        if isinstance(sentence, list):
            embedding = sent_model.encode(sentence[0])
            return embedding.tolist()[0]

        embedding = sent_model.encode(sentence)
        return embedding.tolist()


    # elif type(sentence) == 'str':
    #     print("more then 1")
    #     embedding = sent_model.encode(sentence)
    #     return embedding.tolist()


    else:
        # print("nothing")
        embedding = sent_model.encode("")
        return embedding.tolist()

def node_embedding():
    query = "MATCH (n) return n"

    record = session.run(query)
    # arr = []
    # print(len(record))
    counter = 0

    node_arr = []
    for rec in record:
        counter+=1
        # print(rec)

        # try:
        if rec["n"].labels==frozenset({'Person'}):
            # print(node_arr)
            # print(rec["n"]["name"])
            # print(feature_embedding(str(rec["n"]["name"])))
            # node_arr.append(["Person",feature_embedding(rec["n"]["name"])])
            feat = feature_embedding(rec["n"]["name"], sent_model)

            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
            # rec["n"]["name"] make and store the embedding of it
            # node_arr.append(rec["n"]["name"])

        elif rec["n"].labels==frozenset({'Article'}):
            # print(rec)
            feat = feature_embedding(rec["n"]["title"],sent_model)
            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
                
            # node_arr.append(rec["n"]["abstract"])
        elif rec["n"].labels==frozenset({'Photo'}):
            # node_arr.append(["Photo",feature_embedding(rec["n"]["caption"])[0]])
            feat = feature_embedding(rec["n"]["caption"],sent_model)

            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
            # node_arr.append(rec["n"]["caption"])
        elif rec["n"].labels==frozenset({'Author'}):
            # node_arr.append(["Author",[feature_embedding(rec["n"]["name"])[0]]])
            feat = feature_embedding(rec["n"]["name"],sent_model)

            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
            # node_arr.append(rec["n"]["name"])
        elif rec["n"].labels==frozenset({'Topic'}):
            # node_arr.append(["Topic",feature_embedding(rec["n"]["name"])])
            feat = feature_embedding(rec["n"]["name"],sent_model)

            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
            # node_arr.append(rec["n"]["name"])
        elif rec["n"].labels==frozenset({'Organization'}):
            # node_arr.append(rec["n"]["name"])
            # node_arr.append(["Organization",feature_embedding(rec["n"]["name"])])
            feat = feature_embedding(rec["n"]["name"],sent_model)

            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
        elif rec["n"].labels==frozenset({'Geo'}):
            feat = feature_embedding(rec["n"]["name"],sent_model)


            # node_arr.append(rec["n"]["name"])
            # node_arr.append(["Geo",feature_embedding(rec["n"]["name"])])
            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])

        elif rec["n"].labels==frozenset({}):
            feat = feature_embedding(rec["n"]["name"],sent_model)
            
            if len(feat) == 384:
                node_arr.append([feat])
            else:
                node_arr.append([feat[0]])
            

        else:
            print("problem")

        # except:
        #     print("type of node is not recognized")
    #print(node_arr)
    # print(counter)
    node_arr = np.array(node_arr)

    return node_arr

def edge_embedding():
    query = "Match (n)-[r]->(m) Return n,r,m"

    record = session.run(query)

    link_arr = [[],[]]
    for rec in record:
        #print(rec)
        #print(rec["n"].element_id)
        #print(rec["r"].element_id)
        #print(rec["m"].element_id.split(":")[-1])
        #print(rec["r"].type) 
        link_arr[0].append(int(rec["n"].element_id.split(":")[-1]))
        link_arr[1].append(int(rec["m"].element_id.split(":")[-1]))

    #print(link_arr)
    #print(len(link_arr[0]))
    #print(len(link_arr[1]))
    return link_arr

def reverse2DList(inputList):

    #reverese items inside list
    inputList.reverse()
   
    #reverse each item inside the list using map function(Better than doing loops...)
    inputList = list(map(lambda x: x[::-1], inputList))
    
    #return
    return inputList


def preparing_data(node_arr,link_arr):

    reverse_link_arr = reverse2DList(link_arr)
    
    double_link_arr = [[],[]]

    for i in range(len(link_arr[0])):
        #print(link_arr[0][i])
        double_link_arr[0].append(link_arr[0][i])
        double_link_arr[1].append(link_arr[1][i])

    for i in range(len(reverse_link_arr[0])):
        #print(reverse_link_arr[0][i])
        # print(link_arr)
        double_link_arr[0].append(reverse_link_arr[0][i])
        double_link_arr[1].append(reverse_link_arr[1][i])
    nnode_arr = np.squeeze(node_arr)
    edge_label = [1.]*len(link_arr[0])
    return double_link_arr, nnode_arr, edge_label

    


def preprocessing_graph():
    node_arr = node_embedding()
    link_arr = edge_embedding()

    double_link_arr,nnode_arr, edge_label = preparing_data(node_arr,link_arr)
    #print("here")
    edge_label_index = torch.tensor(link_arr, dtype=torch.long)
    edge_index = torch.tensor(double_link_arr, dtype=torch.long)


    edge_label = torch.tensor(edge_label, dtype=torch.float)


    x = torch.tensor(nnode_arr, dtype=torch.float)
    #print("here")
    
    # x.device
    # edge_index.device
    # edge_label_index.device
    # x.device
    # edge_label_index.device
    # edge_index.device
    data = Data(x=x, edge_index=edge_index, edge_label_index = edge_label_index, edge_label=edge_label)
    
    return data

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # print()

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        #print("save")
        torch.save(x, "conv1.pt")

        #print("save")
        x = self.conv2(x, edge_index)
        torch.save(x, "conv2.pt")
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class loadNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # #print()

    def encode(self, x, edge_index):
        # x = self.conv1(x, edge_index).relu()
        #print("save")
        x = torch.load("models/conv1.pt")

        #print("save")
        # x = self.conv2(x, edge_index)
        x = torch.load("models/conv2.pt")
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0).nonzero(as_tuple=False).t()

# how link prediction is working

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    
    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1))

    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)

    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


# @torch.no_grad()
# def test(data):
#     model.eval()
#     z = model.encode(data.x, data.edge_index)
#     out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
#     return roc_auc_score(data.edge_label.cuda.numpy(), out.cuda().numpy())

    # how to search if cant iterate
    # search until article is not found
    # search in neighbour
    
    # if rec["n"].labels != frozenset({'Article'}):
        
    #     # dfs
    #     # dfs(rec)


    #     pass



def searching(args,final_edge_index):
    # args.id = 3
    num = final_edge_index[0].numpy()
    #print(num)
    arr_num = np.where(num == int(args.id))
    # arr_num  = num.index(args.id)
    # master = np.array([1,2,3,4,5])
    # search = np.array([0])
    # arr_num = np.searchsorted(num, search)
    #print(arr_num)
    for i in range(len(arr_num)):
        new_arr_num = final_edge_index[1][arr_num[i]]
    #print("new_arr_num",new_arr_num)
    # query = "MATCH (n) return n"

    query = "Match (n)-[r]->(m) Return n,r,m"


    record = session.run(query)
    new_arr_num = new_arr_num.numpy()

    #print(new_arr_num[2])

    counter= 0 
    prev = -1
    for rec in record:
        for i in range(len(new_arr_num)):
            # print("rec",rec["n"].element_id.split(":")[-1])
            # print("new_arr_num",new_arr_num[i])

            if rec["n"].element_id.split(":")[-1] == str(new_arr_num[i]):
                if prev != str(new_arr_num[i]):
            # we can go to the nearest article 
                    if rec["n"].labels==frozenset({'Article'}):
                        counter+=1
                        print()
                        print()
                        print(counter,"Article  ")    
                        print(rec["n"]["title"]) 
                        # print(rec) 
                        print(rec["n"]["source_url"] )
                        prev = str(new_arr_num[i])
    
    print() 
    print()
    print()
    return arr_num

def recommendation(args):
    data = preprocessing_graph()
    # if args.training == True:
    num_epoch = 3000
    tick = False
    if tick:
        model = Net(data.x.shape[1], 128, 64)
    else:
        model = loadNet(data.x.shape[1], 128, 64)

    # print("before")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # print("training")
    best_val_auc = final_test_auc = 0
    # if tick:
    if args.training == "True":
        for epoch in range(1, num_epoch):
            loss = train(model,data, optimizer, criterion)
            # val_auc = test(val_data)
            # test_auc = test(test_data)
            # if val_auc > best_val_auc:
            #     best_val = val_auc
            #     final_test_auc = test_auc
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            #       f'Test: {test_auc:.4f}')

        print(f'Final Test: {final_test_auc:.4f}')


    # test
    z = model.encode(data.x, data.edge_index)
    #print("reached")
    final_edge_index = model.decode_all(z)
    #print(final_edge_index)
    searching(args, final_edge_index)
    # print()

def main():
    args = parser.parse_args()
    recommendation(args)


if "__main__" == __name__:
    main()
    print("finished")

# Have to give the command on command line and get the results
