### 配置实验环境
from _common import *
from transformers import TextStreamer

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


def baichuan7b_f_infer(args: baichuan7b_f, test_dataset):
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
    ## 导入Dataset
    import csv
    test_dataset = make_dataset('test', lambda system, user, assistant: 
                                {'system': system, 'user': user, 'assistant':assistant})
    outputs = baichuan7b_f_infer(baichuan7b_f, test_dataset[0])
    with open('test.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)
        for item in outputs:
            csvwriter.writerow([item])