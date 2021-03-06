import os

#import multiprocessing as mp

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

event_prefix = 'event000001001'
hits = load_event('/home/ec2-user/test/event000001001',parts=['hits'])[0]

path = {}
for ind_id in range(hits.count()[0]):
    path[ind_id] = []
for ind_id in range(hits.count()[0]):
#def proc_hit_id(ind_id):
    h_id = hits.iloc[ind_id]
    neigh = []
    if ind_id-1500 < 0:
        lowv = 0
        highv = 3000
    elif ind+1500 > hits.count()[0]-1:
        highv = hits.count()[0]-1
        lowv = highv - 3000
    else:
        lowv = ind_id - 1500
        highv = ind_id + 1500
    for ind in range(lowv,highv):
        cr = hits.iloc[ind]
        dist = max([abs(h_id.x-cr.x),abs(h_id.y-cr.y),abs(h_id.z-cr.z)])
        if len(neigh) < 11:
            neigh.append((dist,ind))
        else:
            neigh.sort()
            if ind % 1000 == 0:
                print((ind_id,ind))
            if (dist < neigh[10][0]):
                neigh[10] = (dist,ind)
                #print("\n",neigh,"\n")
    base = hits.iloc[neigh[0][1]]
    tol = 1e-12
    tolab = 0.3
    sel = []
    for ai in range(1,11):
        for bi in range(1,11):
            if ai == bi:
                continue
            prv = hits.iloc[neigh[ai][1]] #previous
            nxt = hits.iloc[neigh[bi][1]] #next
            d_prv = base-prv              #previous vector
            d_nxt = nxt-base              #next vector
            a = np.array([d_prv.x,d_prv.y])
            b = np.array([d_nxt.x,d_nxt.y])
            aph = 0.1
            bet = 0.01
            while (abs(aph) > tol and abs(bet) > tol and abs(aph/bet-1) > tolab):
                bufa = np.sqrt(1/(aph**2*(a[0]**2+a[1]**2))-0.25)
                bufb = np.sqrt(1/(bet**2*(b[0]**2+b[1]**2))-0.25)
                G = np.array([[a[0]*bufa,-b[0]*bufb],
                              [a[1]*bufa,-b[1]*bufb]])
                b = np.array([aph*a[1]/2+bet*b[1]/2,-aph*a[0]/2-bet*b[0]/2])
                sol = np.linalg.solve(G,b)
                aph = sol[0]
                bet = sol[1]
            if (abs(aph) > tol and abs(bet) > tol and aph*bet>0):
                sel.append((aph/bet,ai,bi,aph,bet))
            sel.sort(reverse=True)
    if (len(sel) > 0):
        #print(sel[0])
        path[ind_id].append((sel[0][4],neigh[sel[0][2]][1]))
        path[neigh[sel[0][1]][1]].append((sel[0][3],ind_id))
