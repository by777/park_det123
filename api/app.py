import celery.states as states
from flask import Flask
from flask import url_for, jsonify
from worker import celery

dev_mode = True
app = Flask(__name__)


@app.route('/add/<int:param1>/<int:param2>')
def add(param1: int, param2: int) -> str:
    task = celery.send_task('tasks.add', args=[param1, param2], kwargs={})
    response = f"<a href='{url_for('check_task', task_id=task.id, external=True)}'>check status of {task.id} </a>"
    return response


@app.route('/check/<string:task_id>')
def check_task(task_id: str) -> str:
    res = celery.AsyncResult(task_id)
    if res.state == states.PENDING:
        pass
        # print(res.state)
    else:
        pass
    # print(str(res.result))
    response = f"fish task? <a href='{url_for('fish_task', task_id=task_id, external=True)}'>{task_id} </a>"
    return response + '<br/>' + res.state + '<br/>' + str(res.result)


@app.route('/health_check')
def health_check() -> str:
    return jsonify("OK")


@app.route('/fish_task/<string:task_id>')
def fish_task(task_id):
    res = celery.control.revoke(task_id, terminate=True)
    return str(res)


@app.route('/start_detect')
def send_config(param1=None, param2=None):
    # 有几个work发几个任务
    global task2, task3, task1
    opts = None
    conf_path = r'schedule_full.json'
    import json
    with open(conf_path, encoding='utf-8') as f:
        opts = json.load(f)

    nworker = len(opts)
    for worker in range(nworker):
        task1 = celery.send_task('tasks.start_detect', kwargs={'noWorker': 'worker' + str(worker)}, shadow=str(worker))
        task2 = celery.send_task('tasks.start_detect', kwargs={'noWorker': 'worker' + str(worker)}, shadow=str(worker))
        task3 = celery.send_task('tasks.start_detect', kwargs={'noWorker': 'worker' + str(worker)}, shadow=str(worker))
    response1 = f"<a href='{url_for('check_task', task_id=task1.id, external=True)}'>check status of {task1.id} </a" \
                f"><br/> "
    # response3 = response2 = response1
    response2 = f"<a href='{url_for('check_task', task_id=task2.id, external=True)}'>check status of {task2.id} </a" \
                f"><br/> "
    response3 = f"<a href='{url_for('check_task', task_id=task3.id, external=True)}'>check status of {task3.id} </a" \
                f"><br/> "
    return response1 + response2 + response3


@app.route('/init_config')
def init_config():
    task = celery.send_task('tasks.init_config')
    response = f"<a href='{url_for('check_task', task_id=task.id, external=True)}'>check status of {task.id} </a>"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001')
