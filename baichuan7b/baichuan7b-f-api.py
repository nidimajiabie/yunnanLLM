#coding=utf-8

import ujson as json
from flask import Flask, Response, request
from traceback import print_exc

from _common import *
from baichuan_infer import baichuan7b_f_infer
from my_logger import make_my_logger

class baichuan7b_f:
    # 指定device
    device_ids = [2,3]
    #
    ### 导入Model, Tokenizer
    # Note: 你需要设置CKPT_FPATH的内容, 指向`.bin`文件, 或`.pth`文件
    CKPT_FAPTH = '/DATA/zqfang/modelscope/modelscope/examples/pytorch/llm_agent/runs/baichuan7B/v30-20230820-232412/output_best/pytorch_model.bin'
    LORA_TARGET_MODULES = ['W_pack']
    #
    # model_dir = snapshot_download('baichuan-inc/baichuan-7B', 'v1.0.5')
    model_dir = "/home/fangzhiqing/.cache/modelscope/hub/baichuan-inc/baichuan-7B/"
    model, tokenizer = get_baichuan7B_model_tokenizer(model_dir)
    model.bfloat16()  # Consistent with training
    #
    ### 导入Lora
    LORA_RANK = 8
    LORA_ALPHA = 32
    LORA_DROPOUT_P = 0  # Arbitrary value
    lora_config = LoRAConfig(
        replace_modules=LORA_TARGET_MODULES,
        rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT_P,
        pretrained_weights=CKPT_FAPTH)
    logger.info(f'lora_config: {lora_config}')
    Swift.prepare_model(model, lora_config)


app = Flask(__name__)

logger = make_my_logger(name='baichuan7b-f-api')


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
            'system' : input,
            'user' : instruction
        }
        output = baichuan7b_f_infer(baichuan7b_f, test_dataset)
        resp['output'] = output
    except Exception as e:
        print_exc()
        resp['msg'] = 'exception occured during infer process.'
    
    # 3) logging
    logger.info(resp)

    # 4) response
    return Response(json.dumps(resp, ensure_ascii=False),
                    content_type='application/json')