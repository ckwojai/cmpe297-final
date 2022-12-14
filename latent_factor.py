import pandas as pd
import numpy
import time

train_raw = pd.read_csv('train.dat', header=None,
            names=["uid", "iid", "rating", "ts"], sep='\t', engine='python')
u2irmap = {}
for _, r in train_raw.iterrows():
    uid, iid = int(r.uid), int(r.iid)
    if uid not in u2irmap:
        u2irmap[uid] = {}
    u2irmap[uid][iid] = int(r.rating)

max_iid = 1699 # result from train_raw
users = sorted(list(u2irmap))
uirMatrix = [[0 for j in range(maxiid)] for i in users]

for uid, uimap in u2irmap.items():
    for iid, rating in uimap.items():
        i = users.index(uid)
        uirMatrix[i][iid-1] = rating


def matrix_factorization(R, P, Q, K, steps=300, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T

    for step in range(steps):
        print(f"Step {step+1}/{steps}")
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T


start_time = time.time()

R = numpy.array(uirMatrix)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 3

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)


nP, nQ = matrix_factorization(R, P, Q, K)

nR = numpy.dot(nP, nQ.T)
print("--- %s seconds ---" % (time.time() - start_time))

ratings = list(map(round, ratings))
with open("test_latent.dat" ,'w') as f:
    for r in ratings:
        if r:
            f.write(f"{r}\n")
        else:
            f.write(f"{3}\n")
