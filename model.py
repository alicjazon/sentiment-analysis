#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, datetime
import numpy as np
import tensorflow as tf
import data
import utils
import matplotlib.pyplot as plt
from tensorflow.keras.backend import dot
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

tf.reset_default_graph()

train_data, test_data, word2idx = data.load_data()

train_data = shuffle(train_data)
test_data = shuffle(test_data)

max_children = 7 # Max number of children for node
vocab_size = len(word2idx) # Vocabulary size
embed_dim = 16   # Embedding dimension
label_size = 3 # Number of output classes
epochs = 15 # Number of epoch
lr = 0.01 # Learning rate

words = tf.placeholder(tf.int32, name='words')
children = [tf.placeholder(tf.int32, name='children' + str(j)) for j in range(max_children)]
labels = tf.placeholder(tf.int32, name='labels')

def init_weight(M, N):
    return np.random.randn(M, N) / np.sqrt(M + N)

def post_order_traversal(node, tree, list):
    for i in range(len((tree[node]))):
        post_order_traversal(tree[node][i], tree, list)
    list.append(node)    
    
#INPUT GATE      
Wxi = init_weight(embed_dim, embed_dim) 
Whi = init_weight(embed_dim, embed_dim) 
bi = np.zeros(embed_dim)

#FORGET GATE
Wxf = init_weight(embed_dim, embed_dim) 
Whf = init_weight(embed_dim, embed_dim) 
bf = np.zeros(embed_dim)

#MEMORY CELL
Wxc = init_weight(embed_dim, embed_dim)
Whc = init_weight(embed_dim, embed_dim)
bc = np.zeros(embed_dim)

#OUTPUT GATE
Wxo = init_weight(embed_dim,  embed_dim) 
Who = init_weight(embed_dim,  embed_dim)
bo = np.zeros(embed_dim)

#Softmax weights
U = init_weight(embed_dim, label_size) 
bs = np.zeros(label_size)

c0 = np.zeros(embed_dim)

embeddings = init_weight(len(word2idx), embed_dim)

Wxi = tf.Variable(Wxi.astype(np.float64))
Whi = tf.Variable(Whi.astype(np.float64))
bi = tf.Variable(bi.astype(np.float64))
Wxf = tf.Variable(Wxf.astype(np.float64))
Whf = tf.Variable(Whf.astype(np.float64))
bf = tf.Variable(bf.astype(np.float64))
Wxc = tf.Variable(Wxc.astype(np.float64))
Whc = tf.Variable(Whc.astype(np.float64))
bc = tf.Variable(bc.astype(np.float64))
Wxo = tf.Variable(Wxo.astype(np.float64))
Who = tf.Variable(Who.astype(np.float64))
bo = tf.Variable(bo.astype(np.float64))
U = tf.Variable(U.astype(np.float64))
bs = tf.Variable(bs.astype(np.float64))
c0 = tf.Variable(c0.astype(np.float64))
embeddings = tf.Variable(embeddings.astype(np.float64))


def calculate_gates(x_t, h_t1):      
    i_t = tf.nn.sigmoid( dot(x_t,Wxi) + dot(h_t1,Whi) 
        + bi)
    f_t = tf.nn.sigmoid( dot(x_t,Wxf) + dot(h_t1,Whf) 
        + bf)
    c_t = c0 * f_t + i_t * tf.nn.tanh( dot(x_t,Wxc) + dot(h_t1,Whc)   
        + bc)
    o_t = tf.nn.sigmoid( dot(x_t,Wxo) + dot(h_t1,Who) 
        + bo)   
    tf.assign(c0, tf.squeeze(c_t))
    h_t = o_t * tf.nn.tanh(c0)
    return h_t


def loop(hiddens, i):
    w = tf.gather(words, i)
    c = [tf.gather(children[j], i) for j in range(max_children)]
    h_i = tf.expand_dims(tf.gather(embeddings, w), 0)

    for child in c:
        h_i = tf.cond(
                child > -1,
                lambda: calculate_gates(h_i, hiddens.read(child)),
                lambda: h_i)

    hiddens = hiddens.write(i, h_i)
    i = tf.add(i, 1)
    return hiddens, i

def condition(hiddens, i):
        return tf.less(i, tf.squeeze(tf.shape(words)))
    
    
def build_feed_dict(node):

    t, l, w = node

    nodes_list = []
    post_order_traversal(-1, t, nodes_list)
    nodes_list.pop(-1)

    feed_dict = {
        words: [w[node] for node in nodes_list],
        labels: [l[node] for node in nodes_list],
    }
    for j in range(max_children):
        feed_dict[children[j]] = [t[node][j] if len(t[node]) > j else -1 for node in nodes_list]
    for j in range(max_children):
        feed_dict[children[j]] = [nodes_list.index(f) if f > -1 else -1 for f in feed_dict[children[j]]]

    return feed_dict


hiddens = tf.TensorArray(
        tf.float64,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False
    )


hiddens, _ = tf.while_loop(
        condition,
        loop,
        [hiddens, 0],
        parallel_iterations=1
    )

def simple_predict(data):

    y_pred = []
    y_true = []

    for step, t in enumerate(data):
        feed_dict = build_feed_dict(t)
        pred, y_i = session.run([prediction, labels], feed_dict=feed_dict)
        y_pred.append(pred)
        y_true.append(y_i)

    list_flatten = lambda l: [item for sublist in l for item in sublist]

    y_pred = list_flatten(y_pred)
    y_true = list_flatten(y_true)  

    acc = accuracy_score(y_true, y_pred)

    return acc, y_pred, y_true

def predict(data):

    y_pred = []
    y_true = []
    target_names = ['Neutral', 'Positive', 'Negative']

    for step, t in enumerate(data):
        feed_dict = build_feed_dict(t)
        pred, y_i = session.run([prediction, labels], feed_dict=feed_dict)
        y_pred.append(pred)
        y_true.append(y_i)

    list_flatten = lambda l: [item for sublist in l for item in sublist]

    y_pred = list_flatten(y_pred)
    y_true = list_flatten(y_true)  

    acc = accuracy_score(y_true, y_pred)
    conmat = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)

    return acc, y_pred, y_true, conmat, report


logits = tf.matmul(hiddens.concat(),U) + bs
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
prediction = tf.argmax(logits, axis=1)
included_indices = tf.where(tf.less(labels, 2))
l_sum = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
           logits = tf.gather(logits, included_indices), labels = tf.gather(labels, included_indices)))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss) #Adagrad / Momentum /GradientDdescentOptimizer
regularization_loss = 0.02 * (tf.nn.l2_loss(Wxi) + tf.nn.l2_loss(U))
full_loss = regularization_loss + l_sum

acc = tf.get_variable('acc', [], dtype=tf.float64)
loss = tf.reduce_sum(tf.get_variable('loss', [], dtype=tf.float64))
session = tf.Session()
saver = tf.train.Saver()
session.run(tf.global_variables_initializer())

train_length = sum(1 for _ in train_data)
loss_history = []
test_loss = []
test_accuracy = []
train_acc_history = []

for epoch in range(1, 5 + 1):

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', acc)

    [tf.summary.histogram(v.name.replace(':', '_'), v) for v in tf.trainable_variables()]
    merged = tf.summary.merge_all()
    print('\n\nepoch {}'.format(epoch))

    for step, t in enumerate(train_data):
        feed_dict = build_feed_dict(t)
        summary, loss_value, _ = session.run([merged, full_loss, train_op], feed_dict=feed_dict)
        loss_history.append(loss_value)
        sys.stdout.write('\r{} / {}'.format(step, train_length))
   

    for step2, t2 in enumerate(test_data):
        test_dict = build_feed_dict(t)
        temp_test_loss, _ = session.run([full_loss, train_op],
                                             feed_dict=test_dict)
        test_loss.append(temp_test_loss)
        
    acc, ypr, ytr, conm, report = predict(train_data)
    train_acc_history.append(acc)

    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, session.graph)
    writer.add_summary(summary, epoch)
    sys.stdout.write('\r{} / {}\ttrain acc: {}  / loss: {} '.format(step, train_length, acc, np.mean(loss_history)))
    test_acc, pred, true, conm_test, report_test = predict(test_data) 
    test_accuracy.append(test_acc)                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    print('Test Acc: {} \ Test loss {}'.format(test_acc, np.mean(temp_test_loss)))
    sys.stdout.flush()

saver.save(session, 'models/model.ckpt')
utils.plot_confusion_matrix(conm_test, ['Neutralne', 'Pozytywne', 'Negatywne'] )
print(report_test)
utils.subplot_metrics(range(1, 20 + 1), train_acc_history, test_accuracy, 'Zbiór uczący', 'Zbiór testowy', 'Dokładność uczenia')

plt.plot(test_loss)
plt.ylabel('Wartość funkcji kosztu')
plt.show();
utils.plot(range(1, 20 + 1), train_acc_history, 'Dokładność uczenia')


sentence = [['minusem', 'może', 'być', 'jego', 'popularność', ',', 'chociaż', 'mi', 'to', 'nie', 'przeszkadza', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['2 0 2 5 3 7 3 11 11 11 7 2']
parents = data.parent_dict(parents)
lab = [[2, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 


sentence = [['opakowanie', 'pasuje', 'do', 'zapachu', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['2 0 2 3 2']
parents = data.parent_dict(parents)
lab = [[0, 0, 0, 0, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['dla', 'mnie', 'nieodpowiedni', 'model']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['4 1 4 0']
parents = data.parent_dict(parents)
lab = [[0, 0, 2, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['fantastyczny', 'zapach', ',', 'napełniający', 'optymizmem', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['2 0 4 2 4 2']
parents = data.parent_dict(parents)
lab = [[1, 0, 0, 1, 1, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['buteleczka', 'śliczna', ',', 'ale', 'to', 'nie', 'rekompensuje', 'trwałości', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['4 1 4 0 4 7 5 7 4']
parents = data.parent_dict(parents)
lab = [[1, 1, 0, 0, 2, 0, 2, 1, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['flakon', 'dość', 'ładny', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['0 3 1 1']
parents = data.parent_dict(parents)
lab = [[0, 0, 1, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['jest', 'odrobinę', 'za', 'ciężki', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['0 1 2 3 1']
parents = data.parent_dict(parents)
lab = [[0, 2, 2, 2, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['dla', 'mnie', 'zalatuje', 'cytrusowo', '-', 'kwiatowym', 'płynem', 'do', 'mycia', 'szyb', '.']]
data.add_to_word2idx(sentence, word2idx)
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['3 1 0 5 7 5 3 7 8 9 3']
parents = data.parent_dict(parents)
lab = [[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['może', 'trochę', 'ciężki', 'na', 'początku', ',', 'ale', 'po', 'czasie', 'przełamuje', 'go', 'słodycz', 'i', 'staje', 'się', 'subtelniejszy', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['7 3 7 3 4 7 0 10 8 13 10 10 7 13 14 14 7']
parents = data.parent_dict(parents)
lab = [[0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['mam', 'wątpiwości', ',', 'czy', 'się', 'rozwija', ',', 'odbieram', 'go', 'jako', 'dość', 
             'jednowymiarowy', ',', 'ale', 'na', 'tyle', 'przyjemny', 'już', 'przy', 'pierwszym', 
             'zetknieciu', ',', 'że', 'nie', 'przeszkadza', 'mi', 'to', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['14 1 14 6 6 14 10 10 8 6 12 10 10 0 14 15 15 17 17 21 19 23 17 25 23 25 25 14']
parents = data.parent_dict(parents)
lab = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['staniki', 'są', 'słabe']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['2 0 2']
parents = data.parent_dict(parents)
lab = [[0, 0, 2]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

sentence = [['mnie', 'od', 'niego', 'mdli', 'i', 'głowa', 'boli', '.']]
sentence = [[word2idx[word] for word in s] for s in sentence]
parents = ['4 4 2 5 0 7 5 5']
parents = data.parent_dict(parents)
lab = [[0, 0, 0, 2, 0, 0, 0, 0]]
test1 = list(zip(parents, lab, sentence))
test_dict = build_feed_dict(t)
test_acc, pred, true = simple_predict(test1)
print('Accuracy: {} \ Predicted: {}  \ True: {}'.format(test_acc, pred, true)) 

