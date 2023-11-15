### 配置实验环境
from _common import *
from transformers import TextStreamer


class chatglm2_f:
    # 指定device
    device_ids = [2,3]
    #
    ### 导入Model, Tokenizer
    CKPT_FAPTH = '/DATA/zqfang/modelscope/modelscope/examples/pytorch/fang/runs/chatglm2/v0-20230901-221444/v0-20230901-230126/output_best/pytorch_model.bin'
    LORA_TARGET_MODULES = ['query_key_value']
    #
    model_dir = '/home/fangzhiqing/.cache/modelscope/hub/ZhipuAI/chatglm2-6b'
    model, tokenizer = get_chatglm2_model_tokenizer(model_dir)
    model.bfloat16()  # Consistent with training
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

def chatglm2_f_infer(args: chatglm2_f, test_dataset):
    # 指定device
    if torch.cuda.is_available():
        # 初始化 CUDA
        torch.cuda.init()
    else:
        print("CUDA 不可用")
    select_device(args.device_ids)
    #
    ### 推理
    streamer = TextStreamer(args.tokenizer, skip_prompt=True, skip_special_tokens=True)
    d = test_dataset
    system = d['system']
    user = d['user']
    input_ids = tokenize_function(system, user, None, args.tokenizer)['input_ids']
    # print(f'[TEST]{args.tokenizer.decode(input_ids)}', end='')
    input_ids = torch.tensor(input_ids)[None].cuda()
    attention_mask = torch.ones_like(input_ids)
    generate_ids = args.model.generate(input_ids=input_ids, max_new_tokens=512,
                                attention_mask=attention_mask,
                                streamer=streamer, pad_token_id=args.tokenizer.eos_token_id, 
                                temperature=0.7, top_k=50, top_p=0.7, do_sample=True)
    # print()
    # print('-----------------------------------------------------------------------------------')
    output = args.tokenizer.decode(generate_ids[0])
    output = output.split('AI:')[1].replace('</s>','').replace('  ','')
    return output


if __name__ == '__main__':
    ### 导入Dataset
    test_dataset = make_dataset('test', lambda system, user, assistant:
                                {'system': system, 'user': user, 'assistant': assistant})
    outputs = chatglm2_f_infer(chatglm2_f, test_dataset[0])