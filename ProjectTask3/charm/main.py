import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def keysofdict(d):
    rv = []
    for k in d.keys():
        rv.append(k)
    return rv


def lookup(d, l):
    rv = []
    for e in l:
        rv.append(d[e])
    return rv

M = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0]])

M = normalize(M,axis=0, norm='l1')

Ns = 6
s2i = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5}
i2s = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F'}


Nt = 3
t2i = {'rock' : 0, 'jazz' : 1, 'hiphop' : 2}
i2t = {0 : 'rock', 1 : 'jazz', 2 : 'hiphop'}

s2t = {0: {2}, 1: {1,0}, 2: {1,0}, 3: {0}, 4: {1}, 5: {1,2}}
t2s = {0: {1,2,3}, 1:{1,2,4}, 2: {0,5}}

Stag = []
beta = 0.8


S = set()
if(len(Stag) == 0):
    Stag = keysofdict(t2i)

Stagi = lookup(t2i, Stag)

for tag in Stagi:
    S = S.union(t2s[tag])


S = list(S)
p_jump = 1 / len(S)
row_jump = np.repeat(p_jump, Ns)
T = np.zeros([Ns, Ns])
for song in S:
    T[song, :] = row_jump


print('M')
print(M)
print('T')
print(T)

M_sp = sp.csc_matrix(M)
T_sp = sp.csc_matrix(T)

A = beta * M_sp + (1 - beta) * T_sp
A = normalize(A, axis=0, norm='l1')
A = A**100
A = A.toarray()

print('A')
print(A)
print(np.sum(A,0))

print('Done')






data = []
track2id = {}
trackid2tagid = {}
tagid2trackid = {}

for jsondict in data:
    track = jsondict['track_id']
    trackid = track2id[track]
        if trackid in trackid2tagid:
            return 0
        # if not trackid in trackid2tagid:
        #     trackid2tagid[trackid] = set()
        #
        # for tag in jsondict['tags']:
        #     tagid = tag2id[tag[0]]
        #     trackid2tagid[trackid].add(tagid)
        #     if not tagid in tagid2trackid:
        #         tagid2trackid[tagid] = set()
        #     tagid2trackid[tagid].add(trackid)





# n = 5
# A_to_n = np.linalg.matrix_power(A,5)

# f1 = plt.figure()
# ax11 = f1.add_subplot(111)
# sns.heatmap(A, ax=ax11)
#
#
#
# f2 = plt.figure()
# ax21 = f2.add_subplot(111)
# sns.heatmap(A_to_n, ax=ax21)
# plt.show()




