#coding=utf-8

import ujson as json
from flask import Flask, Response, request
from traceback import print_exc

from qwen_infer import *
from my_logger import make_my_logger


app = Flask(__name__)

logger = make_my_logger(name='qwen-f-api')


@app.route('/predict', methods=['POST'])
def predict():
    # 0) define resp json
    resp = {'msg': 'success',
            'instruction': '提取实体和关系',
            'input': '',
            'output': ''
            }
    
    # 1) read json input parameters
    instruction = request.json.get('instruction')
    resp['instruction'] = instruction
    input = request.json.get('input')
    resp['input'] = input
    logger.info({'input': input})
    
    # 2) infer pipeline with exception handle
    try:
        test_dataset = {
            'query': f'{instruction}\n{input}'
        }
        output = qwen_7b_chat_f_infer(qwen_7b_chat_f, test_dataset)
        resp['output'] = output
    except Exception as e:
        print_exc()
        resp['msg'] = 'exception occured during infer process.'
    
    # 3) logging
    logger.info(resp)

    # 4) response
    return Response(json.dumps(resp, ensure_ascii=False),
                    content_type='application/json')