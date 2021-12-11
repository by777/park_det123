# -*- coding: utf-8 -*-
# @TIME : 2021/5/13 10:28
# @AUTHOR : Xu Bai
# @FILE : event_processing.py
# @DESCRIPTION : 上传图片以及上报事件与log
import logging
import time
import os
import requests

switch_offline = True  # 这个开关为了不上报事件，因为上报事件太慢了, 真实环境应为False
if not os.path.exists('logs/'):    os.mkdir('logs/')
if not os.path.exists('pics/'):    os.mkdir('pics/')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='logs/log',
                    filemode='w')
logger = logging.getLogger(__name__)

header = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip,deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'User-Agent': 'Mozilla/5.0',
}


def upload(fileName):
    if switch_offline: return {"code": "SUCCESS_200", "msg": "操作成功",
                               "data": "http://119.45.231.236:8085/preview/event_202105132150579.jpg"}

    upload_url = r"http://119.45.231.236:8085/event/upload"
    img = open(fileName, 'rb')
    # 注意这里一定要设置图片类型
    file = {'file': (fileName, img, 'Content-Type: image/png')}
    try:
        result = requests.post(upload_url, files=file, headers=header)
        logger.info(fileName)  # 本地记录证据
        print('uploadImg: ' + result.text)
        return result.json()
    except Exception as e:
        print(e)
    finally:
        img.close()


def add_event(eventPoint, images):
    if switch_offline:
        return {"code": "SUCCESS_200", "msg": "操作成功", "data": ""}
    # print(images)
    add_event_url = r"http://119.45.231.236:8085/event/addEvent"
    # 2021-03-04 21:40:02
    eventTime = time.strftime('%Y-%m-%d %H:%M:%S')
    eventType = 2
    # eventType = '2021-0513-21-39-10'
    params = {
        'eventTime': eventTime,
        'eventType': eventType,
        'eventPoint': eventPoint,
        'images': [images]

    }
    print(params)
    result = requests.post(add_event_url, json=params, headers=header)
    logger.info(params)
    print('addEvent: ' + result.text)
    return result.json()


if __name__ == '__main__':
    # add_event("2021-03-04 21:40:02",2,"A栋东南角",
    # "https://fuss10.elemecdn.com/a/3f/3302e58f9a181d2509f3dc0fa68b0jpeg.jpeg")
    add_event(eventPoint='1', images='http://119.45.231.236:8085/preview/event_202103091650170.jpg')
