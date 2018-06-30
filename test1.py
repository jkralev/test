import os

import multiprocessing as mp

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

event_prefix = 'event000001001'
hits, cells, particles, truth = load_event(os.path.join('C:\\dan\\parts\\train_100_events', event_prefix))

mem_bytes = (hits.memory_usage(index=True).sum() 
             + cells.memory_usage(index=True).sum() 
             + particles.memory_usage(index=True).sum() 
             + truth.memory_usage(index=True).sum())
print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for ind in range(1000,1040):
    p_id = particles.particle_id[ind]
    t_id = truth.loc[truth.particle_id == p_id]
    ax.plot(t_id.tx,t_id.ty,t_id.tz,color='blue')
    h_ids = hits.iloc[t_id.hit_id-1]
    ax.plot(h_ids.x,h_ids.y,h_ids.z,color='red')

p_id = 22525763437723648
t_id = truth.loc[truth.particle_id == p_id]
h_id = hits.iloc[t_id.hit_id-1]
ax.plot(t_id.tx,t_id.ty,t_id.tz)
ax.plot(h_id.x,h_id.y,h_id.z)

detect = pd.read_csv('C:\\dan\\parts\\detectors\\detectors.csv')
detect_top = [{(7,2),(7,4),(12,10),(12,8),(12,12)},
              {(7,4),(7,2),(7,6),(12,12),(12,10)},
              {(7,6),(7,4),(7,8),(12,10),(12,12),(13,2)},
              {(7,8),(7,6),(7,10),(12,12),(13,2)},
              {(7,10),(7,8),(7,12),(13,2)},
              {(7,12),(7,10),(7,14),(13,2)},
              {(7,14),(7,12),(8,2),(8,4),(8,6),(8,8)}]

ind_id = 8
h_id = hits.iloc[ind_id]
neigh = []
for ind in range(hits.count()[0]):
    cr = hits.iloc[ind]
    dist = max([abs(h_id.x-cr.x),abs(h_id.y-cr.y),abs(h_id.z-cr.z)])
    if len(neigh) < 11:
        neigh.append((dist,ind))
    else:
        neigh.sort()
        if ind % 1000 == 0:
            print(ind)
        if (dist < neigh[10][0]):
            neigh[10] = (dist,ind)
            print("\n",neigh,"\n")

base = hits.iloc[neigh[0][1]]
ax.scatter(base.x,base.y,base.z,color='blue')
for ind in range(1,11):
    cr = hits.iloc[neigh[ind][1]]
    ax.scatter(cr.x,cr.y,cr.z,color='red')

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
            sel.append((aph/bet,ai,bi))
        sel.sort()

path = {}
for ind_id in range(hits.count()[0]):
    path[ind_id] = []
#for ind_id in range(hits.count()[0]):
def proc_hit_id(ind_id):
    h_id = hits.iloc[ind_id]
    neigh = []
    for ind in range(hits.count()[0]):
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
                print("\n",neigh,"\n")
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
        print(sel[0])
        path[ind_id].append((sel[0][4],neigh[sel[0][2]][1]))
        path[neigh[sel[0][1]][1]].append((sel[0][3],ind_id))

pool = mp.Pool(processes=2)
for x in range(hits.count()[0]):
    pool.apply(proc_hit_id, args=(x,))
