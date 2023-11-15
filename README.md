一、baichuan7B
1. 运行目录：'/DATA/zqfang/modelscope/modelscope/examples/pytorch/fang'
2. 微调：基于nation-gpt-ie.json文件进行微调，微调代码见baichuan_sft.py文件
3. 推断：微调代码见baichuan_infer.py文件,给出一个示例如下
# if __name__ == '__main__':
#     # 测试数据
#     testdata = {
#         'instruction': '提取实体和关系',
#         'input': '道教约在南诏时期传入云南，主要分布在昆明、大理、保山、临沧、昭通、丽江等\n地，为汉、白、彝、纳西、瑶等民族部分群众信仰；1950年前，全省有道观40余座，\n'
#     }
#     # 将数据转换为输入的形式
#     instruction = test_dataset['instruction']
#     input = test_dataset['input']
#     test_dataset = {
#         'system' : input,
#         'user' : instruction
#     }
#     # 推断
#     baichuan7b_f_infer(baichuan7b_f, test_dataset[0])
4. 接口
    4.1 激活接口代码见baichuan7b-entrypoint.sh文件
    4.2 调用示例见baichuan7b-f-test.py文件，其中接口地址为http://202.121.140.55:7780/predict
    给出一个示例如下
# CUDA_VISIBLE_DEVICES=2,3 gunicorn -w 1 -b 0.0.0.0:7780 baichuan7b-f-api:app --timeout 300
# import requests
# testdata = {
#    'instruction': '提取实体和关系',
#    'input': '道教约在南诏时期传入云南，主要分布在昆明、大理、保山、临沧、昭通、丽江等\n地，为汉、白、彝、纳西、瑶等民族部分群众信仰；1950年前，全省有道观40余座，\n'
# }
# # 调用baichaun-7B API
# print(requests.post('http://202.121.140.55:7780/predict',
#                      json=testdata,
#                      timeout=300).json())

二、chatglm2-6B
1. 运行目录：'/DATA/zqfang/modelscope/modelscope/examples/pytorch/fang'
2. 微调：基于nation-gpt-ie.json文件进行微调，微调代码见chatglm2_sft.py文件
3. 推断：微调代码见chatglm2_infer.py文件,给出一个示例如下
# if __name__ == '__main__':
#     # 测试数据
#     testdata = {
#         'instruction': '提取实体和关系',
#         'input': '道教约在南诏时期传入云南，主要分布在昆明、大理、保山、临沧、昭通、丽江等\n地，为汉、白、彝、纳西、瑶等民族部分群众信仰；1950年前，全省有道观40余座，\n'
#     }
#     # 将数据转换为输入的形式
#     instruction = test_dataset['instruction']
#     input = test_dataset['input']
#     test_dataset = {
#         'system' : input,
#         'user' : instruction
#     }
#     # 推断
#     chatglm2_f_infer(chatglm2_f, test_dataset[0])
4. 接口
    4.1 激活接口代码见chatglm2-entrypoint.sh文件
    4.2 调用示例见chatgm2-f-test.py文件，其中接口地址为http://202.121.140.55:7790/predict
    给出一个示例如下
# CUDA_VISIBLE_DEVICES=2,3 gunicorn -w 1 -b 0.0.0.0:7790 chatglm2-f-api:app --timeout 300
# import requests
# testdata = {
#    'instruction': '提取实体和关系',
#    'input': '道教约在南诏时期传入云南，主要分布在昆明、大理、保山、临沧、昭通、丽江等\n地，为汉、白、彝、纳西、瑶等民族部分群众信仰；1950年前，全省有道观40余座，\n'
# }
# # 调用chatglm2-6B API
# print(requests.post('http://202.121.140.55:7790/predict',
#                      json=testdata,
#                      timeout=300).json())

三、Qwen-7B-chat
1. 运行目录：'/DATA/zqfang/modelscope2/examples/pytorch/llm/src'
2. 微调：基于nation-gpt-ie.json文件进行微调，微调代码见qwen_sft.py文件
3. 推断：微调代码见qwen_infer.py文件,给出一个示例如下
# if __name__ == '__main__':
#     # 加载模型
#     args, remaining_argv = parse_args(qwen_7b_chat_f)
#     if len(remaining_argv) > 0:
#         if args.ignore_args_error:
#             logger.warning(f'remaining_argv: {remaining_argv}')
#         else:
#             raise ValueError(f'remaining_argv: {remaining_argv}')
#     # 测试数据
#     testdata = {
#         'instruction': '提取实体和关系',
#         'input': '道教约在南诏时期传入云南，主要分布在昆明、大理、保山、临沧、昭通、丽江等\n地，为汉、白、彝、纳西、瑶等民族部分群众信仰；1950年前，全省有道观40余座，\n'
#     }
#     # 将数据转换为输入的形式
#     instruction = test_dataset['instruction']
#     input = test_dataset['input']
#     test_dataset = {
#         'query': f'{instruction}\n{input}'
#     }
#     # 推断
#     qwen_7b_chat_f_infer(args, test_dataset)
4. 接口
    4.1 激活接口代码见qwen-f-entrypoint.sh文件
    4.2 调用示例见qwen-f-test.py文件，其中接口地址为http://202.121.140.55:7800/predict
    给出一个示例如下
# CUDA_VISIBLE_DEVICES=2,3 gunicorn -w 1 -b 0.0.0.0:7800 qwen-f-api:app --timeout 300
# import requests
# testdata = {
#    'instruction': '提取实体和关系',
#    'input': '道教约在南诏时期传入云南，主要分布在昆明、大理、保山、临沧、昭通、丽江等\n地，为汉、白、彝、纳西、瑶等民族部分群众信仰；1950年前，全省有道观40余座，\n'
# }
# # 调用Qwen-7B-chat API
# print(requests.post('http://202.121.140.55:7800/predict',
#                      json=testdata,
#                      timeout=300).json())