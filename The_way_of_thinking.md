# My updating for [gzholtkevych/netcomm](https://github.com/gzholtkevych/netcomm)

* My Python version is
* My Numpy version is 1.16.4
* My NetworkX version is 2.4

My task is to use multyprocess unit for `experiment_script.py` for faster work of this script. Watch this script before reading this.

Firstly, I want to analyze script and find errors.

## Numpy and other flaws in the code

The first error is using `np.random.default_range()`. In version 1.16.4 I got `AttributeError`. Let's see it:
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

`PEP8` says, that users should import standard library's modules before other ones. In that script, function `exit` from module `sys` was imported after `numpy`, `matplotlib` etc. Since I also need `JSON` and `multiprocessing` and don't need `sys` and `matplotlib`, my imports looks like this:

```python
import json
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
I don't need to define `net.edges[channel]['a']` in if/else, because I can define it before if/else construction thus:
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
To open logging file, I'm going to use `with\as`. Opening file with `open()` and then `file.close()` is bad practise, as `PEP8` says. So, file's opening looks like this:
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

I should use module, called multiprocessing to ignore GIL and speed up this script. I want to use `Process()` class, which uses arguments called `target`(function) and `args`(tuple). But I had one problem: I couldn't change global variable like NetworkX graph or simple list, so I defined two `mp.Manager.list` for saving nodes and edges of "net". So, I "removed" NetworkX from this script (see below).

```python
# multiprocessing manager
manager = mp.Manager()

# specify community net
nodes = manager.dict()
edges = manager.dict()
```

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

How you can see, I used `mp.Lock` as argument of each function. I did this, because threads can conflict with each other, and the best way to fix it is to use `mp.Lock` and `Process.join()`. You'll see it below, when I'll show calls of functions with `Process()`.

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




