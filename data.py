#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

def parent_dict(data):
    test_trees = []
    for tree in data:
                test_trees.append(defaultdict(list))
                t = tree.split()
                for i in range(len(t)):
                    word_nb = i 
                    parent_nb = int(t[i]) - 1
                    test_trees[-1][parent_nb].append(word_nb)               
    return test_trees

def add_to_word2idx(data, word2idx):
    for sentence in data:
        for word in sentence:
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx

def load_data():
    
    word2idx = {}
    train_paths = {}
    test_paths = {}
    
    # Load train data
    
    train_paths['labels'] = open('dane/neg_labels.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    
    train_paths['labels'].pop(-1)
    train_paths['labels'].extend(open('dane/jun18_labels.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    train_paths['labels'].pop(-1)
    train_paths['labels'].extend(open('dane/sklad_labels.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    train_paths['labels'].pop(-1)
    train_paths['labels'].extend(open('dane/rev_labels.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))    
    
    train_paths['sentence'] = open('dane/neg_sentence.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    
    train_paths['sentence'].pop(-1)
    train_paths['sentence'].extend(open('dane/jun18_sentence.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    train_paths['sentence'].pop(-1)
    train_paths['sentence'].extend(open('dane/sklad_sentence.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    train_paths['sentence'].pop(-1)
    train_paths['sentence'].extend(open('dane/rev_sentence.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    
    train_paths['parents'] = open('dane/neg_parents.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') 
    
    train_paths['parents'].pop(-1)
    train_paths['parents'].extend(open('dane/jun18_parents.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    train_paths['parents'].pop(-1)
    train_paths['parents'].extend(open('dane/sklad_parents.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    train_paths['parents'].pop(-1)
    train_paths['parents'].extend(open('dane/rev_parents.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n'))
    
    # Load test data
    
    test_paths['labels'] = open('dane/polevaltest_labels.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    test_paths['sentence'] = open('dane/polevaltest_sentence.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    test_paths['parents'] = open('dane/polevaltest_parents.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') 
    
    
    train_paths['sentence'] = [line.lower() for line in train_paths['sentence']]
    test_paths['sentence'] = [line.lower() for line in test_paths['sentence']]
    
    
    train_paths['labels'] = [[int(j) for j in i.split()] for i in train_paths['labels']]
    train_paths['labels'] = [[2 if j == -1 else j for j in i] for i in train_paths['labels']]
    test_paths['labels'] = [[int(j) for j in i.split()] for i in test_paths['labels']]
    test_paths['labels'] = [[2 if j == -1 else j for j in i] for i in test_paths['labels']]
    
    train_paths['parents'] = parent_dict(train_paths['parents'])          
    test_paths['parents'] = parent_dict(test_paths['parents'])
     
    
    
    train_paths['sentence'] = [i.split() for i in train_paths['sentence']]
    test_paths['sentence']  = [i.split() for i in test_paths['sentence']]
        
    sentences = train_paths['sentence']
    add_to_word2idx(sentences, word2idx)
    sentences = [[word2idx[word] for word in sentence] for sentence in sentences]
    
    train_paths['sentence'] = sentences
    
    test_sentences = test_paths['sentence']
    add_to_word2idx(test_sentences, word2idx)
    test_sentences = [[word2idx[word] for word in sentence] for sentence in test_sentences]
    
    test_paths['sentence'] = test_sentences
    
    train_data = list(zip(train_paths['parents'], train_paths['labels'], train_paths['sentence']))
    
    test_data = list(zip(test_paths['parents'], test_paths['labels'], test_paths['sentence']))
    train_data.pop(-1)
    test_data.pop(-1)
    return train_data, test_data, word2idx
