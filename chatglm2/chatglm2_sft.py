### 配置实验环境
# !pip install modelscope
# !pip install numpy pandas matplotlib scikit-learn
# !pip install transformers datasets
# !conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# !pip install tqdm tensorboard torchmetrics sentencepiece charset_normalizer accelerate

# !pip install numpy -U  # Resolve torchmetrics dependencies and update numpy

from _common import *
device_ids = [0, 1]
select_device(device_ids)
_ = seed_everything(42)



### 导入Model, Tokenizer
WORK_DIR = 'runs/chatglm2'
LORA_TARGET_MODULES = ['query_key_value']
#
# model_dir = snapshot_download('ZhipuAI/chatglm2-6b', 'v1.0.6')
model_dir = '/home/fangzhiqing/.cache/modelscope/hub/ZhipuAI/chatglm2-6b'
model, tokenizer = get_chatglm2_model_tokenizer(model_dir)
#
GRADIENT_CHECKPOINTING = True
if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()



### 准备Lora
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT_P = 0.1
lora_config = LoRAConfig(
    replace_modules=LORA_TARGET_MODULES,
    rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT_P)
logger.info(f'lora_config: {lora_config}')
Swift.prepare_model(model, lora_config)
#
show_freeze_layers(model)
print_model_info(model)
_p = list(model.parameters())[100]
logger.info(f'device: {_p.device}, dtype: {_p.dtype}')
model.bfloat16()



### 导入Dataset
tokenize_function = partial(tokenize_function, tokenizer=tokenizer)
train_dataset = make_dataset('train', tokenize_function)
val_dataset = make_dataset('val', tokenize_function)
# Data analysis
stat_dataset(train_dataset)
stat_dataset(val_dataset)
data_collate_fn = partial(data_collate_fn, tokenizer=tokenizer)
print_examples(train_dataset[0], tokenizer)



### 配置Config
cfg_file = os.path.join(model_dir, 'configuration.json')
#
BATCH_SIZE = 1
MAX_EPOCHS = 1
T_max = get_T_max(len(train_dataset), BATCH_SIZE, MAX_EPOCHS, True)
WORK_DIR = get_work_dir(WORK_DIR)
EVAL_INTERVAL = 200
CONFIG = Config({
    'train': {
        'dataloader': {
            'batch_size_per_gpu': BATCH_SIZE,
            'workers_per_gpu': 1,
            'shuffle': True,
            'drop_last': True,
            'pin_memory': True
        },
        'max_epochs': MAX_EPOCHS,
        'work_dir': WORK_DIR,
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'options': {
                'cumulative_iters': 16, 'grad_clip': {
                    'norm_type': 2,
                    'max_norm': 2.0
                }
            }
        },
        'lr_scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': T_max,
            'eta_min': 1e-5,
            'options': {
                'by_epoch': False,
                'warmup': {
                    'type': 'LinearWarmup',
                    'warmup_ratio': 0.1,
                    'warmup_iters': 200
                }
            }
        },
        'hooks': [
            {'type': 'CheckpointHook', 'by_epoch': False, 'interval': EVAL_INTERVAL, 'max_checkpoint_num': 1},
            {'type': 'EvaluationHook', 'by_epoch': False, 'interval': EVAL_INTERVAL},
            {'type': 'BestCkptSaverHook',
                'metric_key': 'acc',
                'save_best': True, 'rule': 'max', 'max_checkpoint_num': 1},
            {'type': 'TextLoggerHook',
                'by_epoch': True,  # Whether EpochBasedTrainer is used
                'interval': 5},
            {'type': 'TensorboardHook', 'by_epoch': False, 'interval': 5}
        ]
    },
    'evaluation': {
        'dataloader': {
            'batch_size_per_gpu': BATCH_SIZE,
            'workers_per_gpu': 1,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True
        },
        'metrics': [
            {'type': 'my_metric', 'vocab_size': tokenizer.vocab_size}
        ]
    }
})



### 微调
def cfg_modify_fn(cfg: Config) -> Config:
    cfg.update(CONFIG)
    return cfg


trainer = EpochBasedTrainer(
    model=model,
    cfg_file=cfg_file,
    data_collator=data_collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    remove_unused_data=True,
    seed=42,
    cfg_modify_fn=cfg_modify_fn,
)

trainer.train()


# ### 可视化
# tensorboard 命令: (e.g.)  
# 'tensorboard --logdir /home/hackathon/my_git/agent/runs/baichuan/v29-20230816-153524 --port 6006'
from _common import *
tb_dir = os.path.join(WORK_DIR, 'tensorboard_output')
fname = os.listdir(tb_dir)[0]
tb_path = os.path.join(tb_dir, fname)
#
data = read_tensorboard_file(tb_path)
print(data.keys())


savepath = "/DATA/zqfang/modelscope/modelscope/examples/pytorch/fang/runs/chatglm2/v0-20230901-221444/v0-20230901-230126"
plot_image(data, 'loss', 0.9, savepath)
plot_image(data, 'lr', 0, savepath)
plot_image(data, 'evaluation/acc', 0, savepath)
plot_image(data, 'evaluation/loss', 0, savepath)