# Author:Xu Bai
# Contact: 1373953675@qq.com
# Datetime:2021/10/6 下午1:15
# Software: PyCharm
# Desc:
import json

from datetime import datetime

json_path = 'schedule_full.json'

j = json.load(open(json_path))
print(j)
print(dir(j))

nworker = len(j)
print(nworker)

print(j.get('worker1'))
print(j.values())
from multiprocessing import Process, Queue


def f(q):
    q.put([42, None, 'hello'])


if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())  # prints "[42, None, 'hello']"
    p.join()
