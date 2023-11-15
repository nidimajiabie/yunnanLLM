import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # fang
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import BitsAndBytesConfig, GenerationConfig, TextStreamer
from utils import (MODEL_MAPPING, TEMPLATE_MAPPING,
                   get_model_tokenizer, get_preprocess, inference,
                   select_bnb, select_dtype, show_layers)

from swift import Swift, get_logger
from swift.utils import parse_args, print_model_info, seed_everything

from modelscope import MsDataset


logger = get_logger()


@dataclass
class qwen_7b_chat_f:
    model_type: str ='qwen-7b-chat'
    sft_type: str = 'lora'
    template_type = None
    ckpt_dir: str = '/DATA/zqfang/modelscope2/examples/pytorch/llm/src/runs/qwen-7b-chat/v0-20230903-140025/checkpoint-205'
    eval_human: bool = False  # False: eval test_dataset
    #
    seed: int = 42
    dtype: str = 'bf16'
    ignore_args_error: bool = False  # True: notebook compatibility
    #
    system: str = 'you are a helpful assistant!'
    max_length: Optional[int] = 1024
    #
    # quantization_bit: Optional[int] = field(
    #     default=None, metadata={'choices': {4, 8}})
    # bnb_4bit_comp_dtype: str = field(
    #     default=None, metadata={'choices': {'fp16', 'bf16', 'fp32'}})
    # bnb_4bit_quant_type: str = field(
    #     default='nf4', metadata={'choices': {'fp4', 'nf4'}})
    # bnb_4bit_use_double_quant: bool = True
    #
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9
    skip_prompt: Optional[bool] = None
    torch_dtype = None # fang
    # def __post_init__(self):
    #     if not os.path.isdir(self.ckpt_dir):
    #         raise ValueError(f'Please enter a valid ckpt_dir: {self.ckpt_dir}')
    #     if self.template_type is None:
    #         self.template_type = MODEL_MAPPING[self.model_type].get(
    #             'template', 'default')
    #         logger.info(f'Setting template_type: {self.template_type}')
    #     #
    #     self.torch_dtype, _, _ = select_dtype(self.dtype)
    #     if self.bnb_4bit_comp_dtype is None:
    #         self.bnb_4bit_comp_dtype = self.dtype
    #     self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
    #         self.quantization_bit, self.bnb_4bit_comp_dtype)
    #     if self.skip_prompt is None:
    #         self.skip_prompt = self.eval_human
    #        
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(seed)
    #
    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    model, tokenizer = get_model_tokenizer(
        model_type, torch_dtype, **kwargs)
    #
    # ### Preparing lora
    if sft_type == 'lora':
        model = Swift.from_pretrained(model, ckpt_dir)
    #
    show_layers(model)
    print_model_info(model)

def qwen_7b_chat_f_infer(args: qwen_7b_chat_f, test_dataset) -> None:
    #
    # ### Inference
    template_type = MODEL_MAPPING[args.model_type]['template']
    preprocess_func = get_preprocess(
        template_type, args.tokenizer, args.system, args.max_length, batched=False)
    streamer = TextStreamer(
        args.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=args.tokenizer.pad_token_id,
        eos_token_id=args.tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    #
    if args.eval_human:
        while True:
            query = input('<<< ')
            data = {'query': query}
            input_ids = preprocess_func(data)['input_ids']
            output = inference(input_ids, args.model, args.tokenizer, streamer, generation_config,
                      args.skip_prompt)
    else:
        input_ids = preprocess_func(test_dataset)['input_ids']
        output = inference(input_ids, args.model, args.tokenizer, streamer, generation_config,
                    args.skip_prompt)
        output = output.split('assistant\n')[1].replace('<|im_end|><|endoftext|>','')
    return output
