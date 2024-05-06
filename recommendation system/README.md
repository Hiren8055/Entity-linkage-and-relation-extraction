# Recommendation system

## Introduction

In graph we are performing link prediction for recommendatoin system, User -> Articles.


Code it divided into two main structures :

1. Embeddings 
2. Recommendation system




### 1. Embeddings

It takes the data from Neo4j and creates the embedding using GCN, we are performing link prediction for recommendation system which would require embeddings of nodes and relationship.


Usage:
python transR.py [parameters]

Possible parameters includes:

`-d [str]`: Which dataset to use? Possible selections are FB13, FB15k, WN11, WN18.

`-l [float]`: Initial learning rate. Suitable for TransE and TransH. Default 0.001.

`-l1 [float]`: Learning rate for the first phase. Suitable for TransR and TransD. Default 0.001.

`-l2 [float]`: Initial learning rate for the second phase, if -es set to > 0. Suitable for TransR and TransD. Default 0.0005.

`-es [int]`: Number of times for decrease of learning rate. If set to 0, no learning rate decrease will occur. Default 0.

`-L [int]`: If set to 1, it will use L1 as dissimilarity, otherwise L2. Default 1.

`-em [int]`: Embedding size of entities and relations. Default 100.

`-nb [int]`: How many batches to train in one epoch. Default 100.

`-n [int]`: Maximum number of epochs to train. Default 1,000.

`-m [float]`: Margin of margin loss. Default 1.0.

`-f [int]`: Whether to filter false negative triples in training, validating and testing. If set to 1, they will be filtered. Default 1.

`-mo [float]`: Momentum of optimizers. Default 0.9.

`-s [int]`: Fix the random seed, except for 0, which means no random seed is fixed. Default 0.

`-op [int]`: Which optimizer to choose. If set to 0, Stochastic Gradient Descent (SGD) will be used. If set to 1, Adam will be used. Default 1.

`-p [int]`: Port number used by hyperboard. Default 5000.

`-np [int]`: Number of processes when evaluating. Default 4. 


### 2. Recommendation system

It takes the data from Neo4j through Neo4j and python, recommends article according to the user_id.

Usage:
python recommendation.py