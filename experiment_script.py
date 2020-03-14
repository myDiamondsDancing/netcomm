import json
import multiprocessing as mp

import numpy as np
#!!!
from numpy import random as rg
import networkx as nx
import matplotlib.pyplot as plt

from utils import *

DISCLAIMER = -1

# loading nvars, nactors, niter

with open('config.json', 'r') as f:
    config = json.load(f)

nvars = config['nvars']
nactors = config['nactors']
niter = config['niter']
del config


def set_params_actor(n:int, nodes, lock:mp.Lock) -> None:
    '''This function sets parameters for current actor in net.'''

    lock.acquire()

    try:
        params = dict()
        # set parameters of community actors
        params['rho'] = 20
        if n == 0:
            params['choice'] = 0
        else:
            params['choice'] = DISCLAIMER

        # specify initial prefernce densities of community actors
        if n == 0:
            params['w'] = np.array([1.0, 0.0], float)
        elif n == 1:
            params['w'] = np.array([0.0, 1.0], float)
        else:
            params['w'] = uncertainty(nvars) 

        nodes[n] = params
    finally:
        lock.release()


def set_params_channel(channel:tuple, edges, lock:mp.Lock) -> None:
    '''This function sets parameters for current edge in net.'''

    lock.acquire()
    try:
        params = dict()
        alice = channel[0]
        params['a'] = 1.0

        # setting channel dialog matrix
        if alice == 0:
            params['D'] = define_dialogue_matrix(1.0,
                                                 rg.uniform(low=0.2, high=0.6))
        else:
            params['D'] = define_dialogue_matrix(rg.uniform(low=0.2, high=0.6),
                                                 rg.uniform(low=0.2, high=0.6))
        edges[channel] = params
    finally:
        lock.release()


def simulate_dialog(alice:int, bob:int, nodes, edges) -> tuple:
    '''This function simulates dialog between two actors.'''

    # get dialogue matrix of the current dialogue and
    # the preference densities of its participants
    D = edges[(alice, bob)]['D']

    wA = nodes[alice]['w']
    wB = nodes[bob]['w']

    wA_result = np.zeros(nvars)
    wB_result = np.zeros(nvars)

    for v in range(nvars):
        wA_result[v] = D[0, 0] * wA[v] + D[0, 1] * wB[v]
        wB_result[v] = D[1, 0] * wA[v] + D[1, 1] * wB[v]

    return wA_result, wB_result


def simulate_session_ch(channel:tuple, nodes, edges, lock:mp.Lock) -> None:
    '''This function simulates session for current channel.'''
    lock.acquire()

    try:
        # clean auxiliary information
        for actor in channel:
            params = nodes[actor]
            params['result_list'] = list()
            nodes[actor] = params

        # simulate dialogue
        if not Bernoulli_trial(edges[channel]['a']):
            # channel is not active
            return 

        # channel is active
        # determine actors participating as Alice and Bob in the dialogue
        alice, bob = channel

        # simulated dialog
        wA, wB = simulate_dialog(alice, bob, nodes, edges)

        # setting 'w' for each actor
        alice_node = nodes[alice]
        alice_node['result_list'].append(wA)
        nodes[alice] = alice_node

        bob_node = nodes[bob]
        bob_node['result_list'].append(wA)
        nodes[bob]= bob_node
    finally:
        lock.release()


def compute_previous_result(n:int, nodes, edges, lock:mp.Lock) -> None:
    '''This function computes previous result for current actor.'''
    lock.acquire()

    try:
        if nodes[n]['result_list']:
            # actor participates at least in one dealogue
            ndialogues = len(nodes[n]['result_list'])
            w = np.zeros(nvars)
            #!!!
            params = nodes[n]

            for wc in nodes[n]['result_list']:
                np.add(w, wc, w)
                np.multiply(w, 1.0 / ndialogues, params['w']) 

            nodes[n] = params  
    finally:
        lock.release()   


def simulate_session(nodes, edges) -> None:
    '''This function simulates session.'''

    lock = mp.Lock()

    procs = list()
    for channel in edges:
        proc = mp.Process(target=simulate_session_ch, args=(channel, nodes, edges, lock))

        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


    procs = list()
    # compute the previous session result for each community actor
    for n in nodes:
        proc = mp.Process(target=compute_previous_result, args=(n, nodes, edges, lock))

        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def pol_simulation(n:int, nodes, lock:mp.Lock) -> None:
    hn = h(nodes[n]['w'])

    if Bernoulli_trial(np.power(hn, nodes[n]['rho'])):
        # actor 'n' disclaims a choice
        params = nodes[n]
        params['choise'] = DISCLAIMER
        nodes[n] = params
    else:
        # actor 'n' chooses
        params = nodes[n]
        params['choise'] = rg.choice(nvars, 
                                     p=nodes[n]['w'])
        nodes[n] = params


def observation(nodes, edges, lock:mp.Lock) -> tuple:

    procs = list()
    for n in nodes:
        proc = mp.Process(target=pol_simulation, args=(n, nodes, lock))

        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


    # compute average preference density
    W = np.zeros(nvars)

    for n in nodes:
        np.add(W, nodes[n]['w'], W)

    np.multiply(W, 1.0 / nactors, W)

    # compute polling result
    DP = len([1 for n in nodes if nodes[n]['choice'] == DISCLAIMER])

    if DP == nactors:
        # all community actors disclaimed a choice 
        return W, 1.0, uncertainty(nvars)

    NP = nactors - DP
    WP = [None] * nvars

    for v in range(nvars):
        WP[v] = len([1 for n in nodes if nodes[n]['choice'] == v]) / NP

    DP /= nactors
    return W, DP, WP


if __name__ == '__main__':

    # multiprocessing manager
    manager = mp.Manager()

    # specify community net
    nodes = manager.dict()
    edges = manager.dict()

    lock = mp.Lock()

    # set parameters of community actors
    procs = list()
    for n in range(nactors):
        proc = mp.Process(target=set_params_actor, args=(n, nodes, lock))

        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
        

    # set parameters of community channels
    procs = list()
    for i in range(nactors):
        for j in range(i + 1, nactors):
            proc = mp.Process(target=set_params_channel, args=((i, j), edges, lock))

            procs.append(proc)
            proc.start()

    for proc in procs:
        proc.join()


    niter = 10  # define number of iterations

    # set up the experiment
    protocol = [observation(nodes, edges, lock)]
    for istep in range(niter):
        simulate_session(nodes, edges)
        protocol.append(observation(nodes, edges, lock))

    # ----------------------------------------------------------
    # store the experiment outcomes
    # ----------------------------------------------------------
    with open("protocol.dat", "w") as out_file:
        # out_file.write(str(nvars) + "\n")

        for item in protocol:
            for val in item[0]:
                out_file.write(str(val) + " ")

            out_file.write(str(item[1]))

            for val in item[2]:
                out_file.write(" " + str(val))

            out_file.write("\n")
