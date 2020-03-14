# My updating for [gzholtkevych/netcomm](https://github.com/gzholtkevych/netcomm)

* My Python version is
* My Numpy version is 1.16.4
* My NetworkX version is 2.4

My task is to use multyprocess unit for `experiment_script.py` for faster work of this script.
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


