# My updating for [gzholtkevych/netcomm](https://github.com/gzholtkevych/netcomm)

* My Python version is
* My Numpy version is 1.16.4
* My NetworkX version is 2.4

My task is to use multyprocess unit for `experiment_script.py` to make script work faster. Watch this script before reading this.

Firstly, I want to analyze script and find errors.

## Numpy and other flaws in the code

The first error is using `np.random.default_range()`. In version 1.16.4 I got `AttributeError`. Let's look at it:
```python
import numpy as np

rg = np.random.default_range()
```

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-4-83c2ec738e1c> in <module>
      1 import numpy as np
      2 
----> 3 rg = np.random.default_range()

AttributeError: module 'numpy.random' has no attribute 'default_range'
```

Now function uniform is in `np.random`. So, I'll use `np.random.uniform`.
Also, I don't want to use `rg = np.random`. Look down:

```python
from numpy import random as rg
```

`PEP8` says, that users should import standard library's modules before other ones. In that script, function `exit` from module `sys` was imported after `numpy`, `matplotlib` etc. Since I also need `JSON`, `pickle` and `multiprocessing` and don't need `sys` and `matplotlib`, my imports looks like this:

```python
import json
import pickle
import multiprocessing as mp

import numpy as np
from numpy import random as rg
import networkx as nx

from utils import *
```

Another problem is tuple unpacking. Line 81:
```python
alice, bob = min(channel), max(channel)
```
Since `net.edges()` returns tuples like (a, b), where a < b, I can define `alice` and `bob` like this:
```python
alice, bob = channel
```
In setting parameters for channels, in lines 29-41, we had another problem:
```python
# set parameters of community channels
for channel in net.edges:
    alice = min(channel)
    if alice == 0:
        net.edges[channel]['a'] = 1.0
        net.edges[channel]['D'] = define_dialogue_matrix(
            1.0,
            rg.uniform(low=0.2, high=0.6)
        )
    else:
        net.edges[channel]['a'] = 1.0
        net.edges[channel]['D'] = define_dialogue_matrix(
            rg.uniform(low=0.2, high=0.6),
            rg.uniform(low=0.2, high=0.6)
        )
```
I don't need to define `net.edges[channel]['a']` in if/else, because I can define it before if/else construction, thus:
```python
# set parameters of community channels
for channel in net.edges:
    alice = min(channel)
    net.edges[channel]['a'] = 1.0
    
    if alice == 0:
        net.edges[channel]['D'] = define_dialogue_matrix(
            1.0,
            rg.uniform(low=0.2, high=0.6)
        )
    else:
        net.edges[channel]['D'] = define_dialogue_matrix(
            rg.uniform(low=0.2, high=0.6),
            rg.uniform(low=0.2, high=0.6)
        )
```
To open logging file, I'm going to use `with\as`. Opening file with `open()` and then `file.close()` is bad practice, as `PEP8` says. So, file's opening looks like this:
```python
with open("protocol.dat", "w") as out_file:
    # out_file.write(str(net.nvars) + "\n")
    for item in protocol:
        for val in item[0]:
            out_file.write(str(val) + " ")
        out_file.write(str(item[1]))
        for val in item[2]:
            out_file.write(" " + str(val))
        out_file.write("\n")
```

## Multiprocessing

I should use module, called multiprocessing to ignore GIL and speed up the script. I want to use `Process()` class, which uses arguments called `target`(function) and `args`(tuple). But I had one problem: I couldn't change global variable like NetworkX graph or simple list, so I defined two `mp.Manager.list` for saving nodes and edges of "net". So, I "removed" NetworkX from this script (see below).

```python
# multiprocessing manager
manager = mp.Manager()

# specify community net
nodes = manager.dict()
edges = manager.dict()
```

> I couldn't even change dicts in this lists, so to change parameters for channel or actor I copy its parameters set (dict) into variable, change it and then set parameters by appending this variable to list. 

For using `Process()` I needed to transform all code snippets into functions. I began from setting parameters for each channel and actor:
```python
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
```

How you can see, I used `mp.Lock` as argument of each function. I did this, because threads can conflict with each other, and the best way to fix it is to use `mp.Lock` and `Process.join()`. You can see it below, when I show calls of functions with `Process()`.

Then I defined function for simulating dialog between two actors:
```python
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
```

Next I defined two functions, used in simulating session. These are simulating session for current channel and computing previous results for each actor:
```python
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
 ```
 
 Further I defined function for simulating session, where I called these two functions with `Process()`:
 ```python
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
  ```

There is one code snippet in observation, which performed for each actor. I transormed it to separate function, called `pol_simulation`, which used `Bernoulli_trial`:
```python
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
```

So, my function for observation looks like that:
```python
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
```
In this function, I'm calling `pol_simulation` function with `Process()` and calculating necessary values.

It's time to set up the experiment!

```python
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


    # set up the experiment
    protocol = [observation(nodes, edges, lock)]
    for istep in range(niter):
        simulate_session(nodes, edges)
        protocol.append(observation(nodes, edges, lock))
```
Then, I need to write result to file (I told about that above):
```python
    with open(output_path, "w") as out_file:
        # out_file.write(str(nvars) + "\n")

        for item in protocol:
            for val in item[0]:
                out_file.write(str(val) + " ")

            out_file.write(str(item[1]))

            for val in item[2]:
                out_file.write(" " + str(val))

            out_file.write("\n")
```

## Control script with JSON-file

I don't like defining variables with important values in this script, like `niters` or `output_path`. So, I decided to creare JSON-file with next fields:
* nactors
* niter
* nvars
* output_path
* net_output_path

In the beggining of this script, I'm opening this file and loading all values I need:
```python
# loading nvars, nactors, niter etc.
with open('config.json', 'r') as f:
    config = json.load(f)

nvars = config['nvars']
nactors = config['nactors']
niter = config['niter']
output_path = config['output_path']

# we already don't need this dict 
del config
```

## Creating NetworkX graph

It is important to create NetworkX model for further work. So, script creates complete graph with number of nodes equal to nactors and copies every field in parameters set in `nodes` and `edges` to `net.nodes` and `net.edges`:
```python
    net = nx.complete_graph(nactors)

    for n in nodes:
        for p in nodes[n]:
            net.nodes[n][p] = nodes[n][p]

    for edge in edges:
        for p in edges[edge]:
            net.edges[edge][p] = edges[edge][p]
```
Also, script saves this net to file, `using pickle.dump`:
```python
    with open(net_output_path, 'wb') as f:
        pickle.dump(net, f)
```

## Conclusion

How you can see, I modified script for using multiprocessing. Also, I created docstring for each function, fixes some code problems and realised some new things.

Thanks for attention, mydiamondsdancing.







