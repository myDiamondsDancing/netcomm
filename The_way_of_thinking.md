# My updating for [gzholtkevych/netcomm](https://github.com/gzholtkevych/netcomm)

*My Python version is
*My Numpy version is 1.16.4
*My NetworkX version is 2.4

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

