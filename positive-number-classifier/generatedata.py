#!/usr/bin/env python3

"""Generate "data.csv", which is used as classifier training data.

The CSV file is formatted like:
value1,value2,positive
-3.12,4.23,True
-5,5,False
-4,-4.1,False
...
"""

import random

with open('data.csv', 'w') as f:
    f.write('value1,value2,positive\n')
    for _ in range(1000000):
        value1 = random.uniform(-1000, 1000)
        value2 = random.uniform(-1000, 1000)
        f.write('%s,%s,%s\n' % (value1,value2,(value1+value2)>0))