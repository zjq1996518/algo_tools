import asyncio
import math
import pathlib

import aiohttp
import cv2
import requests

OCR_URL = 'http://ocr.yxuer.com/predict/ocr_system2'


def ocr_by_url(img_url, batch_size=4):
    results = []
    # 接口传送的参数
    if isinstance(img_url, str):
        img_url = [img_url]

    for i in range(math.ceil(len(img_url) // batch_size)):
        batch = img_url[i*batch_size:(i+1)*batch_size]
        data = {
            "images": batch,
            "type": 'ocr', "box_thresh": 0.45}
        r = requests.post(url=OCR_URL, json=data, timeout=3600)
        # 发送请求
        results += r.json()['results']

    return results


def ocr_by_img(img, type='ocr-page'):
    # 测试的接口url
    # 接口传送的参数
    headers = {
        'Content-Type': 'application/octet-stream',
        'type': type,
        'box_thresh': '0.4',
        'ratio': '1',
        'model_type': 'light'
    }
    img = cv2.imencode('.jpg', img)[1]
    img = img.tobytes()
    r = requests.post(url=OCR_URL, headers=headers, data=img)
    # 发送请求
    ocr_rst = r.json()['results'][0]
    return ocr_rst


def async_ocr_by_img(imgs, type):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_async_ocr_by_img(imgs, type))


async def _async_ocr_by_img(imgs, type):
    if not isinstance(imgs, list):
        imgs = [imgs]
    path_obj = pathlib.PurePath(OCR_URL)
    parts = path_obj.parts
    base_url = f'{parts[0]}//{parts[1]}'

    headers = {
        'Content-Type': 'application/octet-stream',
        'type': type,
        'box_thresh': '0.4',
        'ratio': '1',
        'model_type': 'light'
    }
    tasks = []
    url = f'/{parts[-2]}/{parts[-1]}'
    async with aiohttp.ClientSession(base_url, headers=headers) as session:
        for img in imgs:
            img = cv2.imencode('.jpg', img)[1]
            img = img.tobytes()
            tasks.append(_request(session, url, img))
        rsts = await asyncio.gather(*tasks)
    return rsts


async def _request(session, url, data):
    async with session.post(url, data=data) as resp:
        rst = await resp.json()
        content = rst['results'][0]
        content = content[0] if len(content) > 0 else ''
        return content
