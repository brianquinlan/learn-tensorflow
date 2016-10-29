#!/usr/bin/python3

# 5a + 2b - 3c = d
import random

with open('linear.csv', 'w') as f:
    f.write('a,b,c,d\n')
    for _ in range(1000000):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        c = random.uniform(-10, 10)
        d = 5 * a + 2 * b - 3 * c
        f.write('%s,%s,%s,%s\n' % (a,b,c,d))