import os
import time
from celery.schedules import timedelta, crontab
from datetime import timedelta
from celery import Celery

from yolov5_DeepSort.getArgs import start_detect_job  #
from celery.task import periodic_task
import json

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

CELERY_IMPORTS = 'celery-queue.tasks'
CELERY_TIMEZONE = 'Asia/Shanghai'
# 单个任务的最大运行时间
CELERY_TASK_TIME_LIMIT = 12 * 30
# 每个work最多100个任务被销毁
CELERY_MAX_TASK_PER_CHILD = 100
CELERY_DEFAULT_QUEUE = 'work_queue'
CELERY_FORCE_EXECV = True
CELERY_CONCURRENCY = 4  # 并发的work数量的
CELERY_ACKS_LATE = True  # 允许充实

CELERY_QUEUES = {
    'beat_tasks': {
        'exchange': 'beat_tasks',
        'exchange_type': 'direct',
        'binding_key': 'beta_tasks'
    },
    'work_queue': {
        'exchange': 'work_queue',
        'exchange_type': 'direct',
        'binding_key': 'work_queue'
    }
}
# 定时任务的描述
CELERYBEAT_SCHEDULE = {
    "add": {
        "task": "tasks.add",
        "schedule": timedelta(seconds=1),
        'args': (1600, 16),
        'options': {
            'queue': 'beat_tasks'
        }
    },
    "say_hi": {
        "task": "tasks.say_hi",
        "schedule": timedelta(seconds=1),
        "args": "Hello.........",
        'options': {
            'queue': 'beat_tasks'
        }
    }
}


@celery.task(name='tasks.add')
def add(x: int, y: int) -> int:
    time.sleep(3)
    print("result=", x + y)
    return x + y


@celery.task(name='tasks.init_config')
def init_config(conf_path='schedule_full.json'):
    opts = json.load(open(conf_path))
    # print(opts)
    # start_detect_job(opts)
    return opts


'''周期性执行检测任务'''


# @periodic_task(run_every=timedelta(minutes=5), name='tasks.start_detect')
@celery.task(name="tasks.start_detect")
def start_detect(noWorker):
    opts = init_config()
    print(opts)
    print(noWorker)
    print(opts[noWorker])
    start_detect_job(opts)
    return 0


@periodic_task(run_every=timedelta(days=1), name="tasks.say_hi")
def say_hi(word='this is say_hi func.\n'):
    print('Hi !\n')
    return str(time.time())
