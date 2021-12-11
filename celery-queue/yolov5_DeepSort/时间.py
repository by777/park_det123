import random
import time

stime = time.time()

dataset = ['a', 'b', 'c', 'd', 'e']

for i in dataset:
    time.sleep(1)
    if time.time() - stime >= 1:
        print('break')
        dataset = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', ]
        break
    else:
        print(i)
print(dataset)
