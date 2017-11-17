import time
import pandas as pd
import numpy.linalg as la
import numpy as np
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=3)



path = "/Users/oeken/Downloads/train_triplets.txt"
n = 20000
n_test = 100     # size of the test set
epochs = 50      # how many times to go over the whole dataset?
delta = 0.005     # learning rate
lambda1 = 0.0   # regularizer coeff for p
lambda2 = 0.0   # regularizer coeff for q
S = 1000          # batch size
k = 2           # number of latent factors




# finds the corresponding bin value for a given number x
def count_to_bin(a,b):
    bin = 0
    for x in range (b):
        if a>=2**x and a<=((2**(x+1))-1):
            return x+1
    return b

# assigns incremental IDs and creates 2-way dictionaries
def create_dictionary(list):
    dict_1 = {}
    dict_2 = {}
    count = 0;
    for element in list:
        if not element in dict_1:
            dict_1[element] = count
            dict_2[count] = element
            count = count+1
    return dict_1, dict_2






# reads n triplets from given file and returns them in a list
def read_triplets(path, n):
    with open(path) as infile:
        count = 0;
        triplets = []
        for line in infile:
            temp_list=[]
            count = count+1
            temp_list = line.split()
            triplets.append(temp_list)
            if count==n:
                break
    return triplets


# removes triplets containing entities 5 or less occurrences
def elim_five_or_less(triplets):

    while 1 :
        min_listeners = triplets.groupby('users').count().min()[0]
        min_songs = triplets.groupby('songs').count().min()[0]
        triplets = triplets.groupby('users').filter(lambda x: len(x) > 5)
        triplets = triplets.groupby('songs').filter(lambda x: len(x) > 5)
        if min_listeners>5 and min_songs>5:
            break;
    return triplets


# assigns ids to users and songs and binifies the play counts
def get_ids_and_bins(M_raw):
    M = M_raw.copy()
    users = M[:,0] #first column represents the users
    songs = M[:,1] #second column represents the songs
    ud1, ud2 = create_dictionary(users) #gives incremental integer values to all users
    sd1, sd2 = create_dictionary(songs) #gives incremental integer values to all songs

    M[:,2] = [count_to_bin(x , 7) for x in M[:,2]] #bin representation of the counts
    M[:,1] = [sd1[x] for x in M[:,1]] #incremental represantation of the songs
    M[:,0] = [ud1[x] for x in M[:,0]] #incremental represantation of the listeners
    return M, ud1, ud2, sd1, sd2


# given an instance of p finds associated q's with it, and vice-versa. makes use of dicts
def assoc_pq(M):
    p_to_q = dict()
    q_to_p = dict()
    for row in M:
        if row[0] in p_to_q:
            tup = p_to_q[row[0]]
            tup[0].append(row[1])
            tup[1].append(row[2])
        else:
            p_to_q[row[0]] = [[row[1]], [row[2]]]

        if row[1] in q_to_p:
            tup = q_to_p[row[1]]
            tup[0].append(row[0])
            tup[1].append(row[2])
        else:
            q_to_p[row[1]] = [[row[0]], [row[2]]]
    return p_to_q, q_to_p

def error(P, Q, M):
    e = 0
    for row in M:
        e += (row[2] - P[row[0],:].dot(Q[row[1],:]))**2
    e += lambda1 * np.linalg.norm(P)
    e += lambda2 * np.linalg.norm(Q)
    return e, e/M.shape[0]

def rmse(P, Q, M):
    r = 0
    for row in M:
        r += (row[2] - P[row[0],:].dot(Q[row[1],:]))**2
    r = np.sqrt(M.shape[0])
    return r





triplets = read_triplets(path, n)
triplets = pd.DataFrame(triplets, columns=['users','songs','play_counts'])
triplets.head(5)

print(triplets.shape)

triplets_eliminated = elim_five_or_less(triplets)
triplets_eliminated.head(5)

triplets_eliminated.shape

M_raw = np.asarray(triplets_eliminated) #convert list to ndarray
M_raw[:,2] = list(map(int, M_raw[:,2])) #parse the string counts to integer

M, ud1, ud2, sd1, sd2 = get_ids_and_bins(M_raw)
u = len(ud1)  # number of users
s = len(sd1)  # number of songs
n_new = len(M)
print('# of triplets: %d' % (n_new))
print('# of users: %d' % (u))
print('# of songs: %d' % (s))
print('Total rating slots: %d' % (u*s))
print('Data ratio: %.4f' % (n_new/(u*s)))

# synthetic götten uydurma data
# see --> http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
# u = 5
# s = 4
# M = np.array([[0,0,5],
#                 [0,1,3],
#                  [0,3,1],
#                  [1,0,4],
#                  [1,3,1],
#                  [2,0,1],
#                  [2,1,1],
#                  [2,3,5],
#                  [3,0,1],
#                  [3,3,4],
#                  [4,1,1],
#                  [4,2,5],
#                  [4,3,4]])
# n_new = 13
# print('# of triplets: %d' % (n_new))
# print('# of users: %d' % (u))
# print('# of songs: %d' % (s))
# print('Total rating slots: %d' % (u*s))
# print('Data ratio: %.4f' % (n_new/(u*s)))

np.random.seed(100)
all_indices = np.arange(n_new)
np.random.shuffle(all_indices)
test_indices = all_indices[:n_test]
#trai_indices = all_indices[n_test:]
trai_indices = all_indices # uncomment for synthetic data
print('Test set shape', test_indices.shape)
print('Trai set shape', trai_indices.shape)

np.random.seed(110)
P = np.random.rand(u,k) # init P
Q = np.random.rand(s,k) # init Q

print('P shape: ', P.shape)
print('Q shape: ', Q.shape)
errors = [error(P,Q,M)]
print('Initial error: ', errors)

print()

batch_count = int(n_new / S)
print('Train count:', trai_indices.size)
print('Batch size: %d' % S)
print('Batch count: %d' % batch_count)

t = time.process_time()
print('i. E(P,Q) = %.3f, %.3f' % errors[0])

for e in range(epochs):  # how many times to go over the whole dataset
    t_epoch = time.process_time()
    np.random.shuffle(trai_indices)  # permute the training data
    for b in range(batch_count):  # for each batch

        batch = trai_indices[b*S:b*S+S]  # fetch the next training data indices
        M_b = M[batch]
        p_grad = {}
        q_grad = {}

        for row in M_b:  # compute gradients
            p = row[0]
            q = row[1]
            r = row[2]
            p_vec = P[p,:]
            q_vec = Q[q,:]
            term = (r - p_vec.dot(q_vec))

            if p in p_grad:
                p_grad[p] += term * q_vec
            else:
                p_grad[p] = term * q_vec - (lambda1 * p_vec)

            if q in q_grad:
                q_grad[q] += term * p_vec
            else:
                q_grad[q] = term * p_vec - (lambda2 * q_vec)


        for p in p_grad.keys():
            P[p,:] += delta * p_grad[p]

        for q in q_grad.keys():
            Q[q,:] += delta * q_grad[q]

    error_current = error(P, Q, M)
    errors.append(error_current)
    print('%d. E(P,Q) = %.3f, %.3f -- time: %.2f' % (e, error_current[0], error_current[1], time.process_time()-t_epoch))


elapsed_time = time.process_time() - t
print('%d Epochs, Total seconds: %f' % (epochs, elapsed_time))


plt.plot(errors)