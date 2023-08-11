import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import random
import numpy as np

import lib.callbacks as callbacks
from lib.loggers import WandbLogger
from lib.arg_utils import define_args

from lib import NotALightningTrainer
from lib import nomenclature
from lib.forge import VersionCommand

VersionCommand().run()

args = define_args()

args.modalities = [modality for modality in args.modalities if modality.name in args.use_modalities]

wandb.init(project = 'perceiving-depression', group = args.group, entity = 'perceiving-depression')
wandb.config.update(vars(args))

if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

dataset = nomenclature.DATASETS[args.dataset](args = args)
train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)

architecture = nomenclature.MODELS[args.model](args)
model = nomenclature.TRAINERS[args.trainer](args, architecture)

evaluators = [
    nomenclature.EVALUATORS[evaluator_args.name](args, architecture, evaluator_args.args)
    for evaluator_args in args.evaluators
]

wandb_logger = WandbLogger()

checkpoint_callback_best = callbacks.ModelCheckpoint(
    args = args,
    name = ' 🔥 Best Checkpoint Overall 🔥',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'checkpoints/{args.group}:{args.name}/best/',
    save_best_only = True,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}',
)

checkpoint_callback_last = callbacks.ModelCheckpoint(
    args = args,
    name = '🛠️ Last Checkpoint 🛠️',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'checkpoints/{args.group}:{args.name}/last/',
    save_best_only = False,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}',
)

if args.scheduler == "cyclelr":
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer = model.configure_optimizers(lr = args.scheduler_args.base_lr),
        cycle_momentum = False,
        base_lr = args.scheduler_args.base_lr,
        mode = args.scheduler_args.mode,
        step_size_up = len(train_dataloader) * args.scheduler_args.step_size_up, # per epoch
        step_size_down = len(train_dataloader) * args.scheduler_args.step_size_down, # per epoch
        max_lr = args.scheduler_args.max_lr
    )
elif args.scheduler == "onecyclelr":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = model.configure_optimizers(lr = args.scheduler_args.base_lr),
        max_lr = args.scheduler_args.max_lr,
        steps_per_epoch = round(len(train_dataloader) / args.accumulation_steps),
        epochs = args.epochs,
        anneal_strategy = "linear",
    )
else:
    raise NotImplementedError("Support only 'cyclelr' or 'onecyclelr'")

lr_callback = callbacks.LambdaCallback(
    on_batch_end = lambda: scheduler.step()
)

lr_logger = callbacks.LambdaCallback(
    on_batch_end = lambda: wandb_logger.log('lr', scheduler.get_last_lr()[0])
)

if args.debug:
    print("[🐞DEBUG MODE🐞] Removing ModelCheckpoint ... ")
    callbacks = [lr_callback, lr_logger]
else:
    print("[🐞🐞🐞🐞🐞🐞] REMOVING ModelCheckpoint TO SAVE SPACE ... ")
    print("[🐞🐞🐞🐞🐞🐞] WHEN RUNNING FINAL EXPERIMENTS ADD IT BACK!!!!!!")
    callbacks = [
        # checkpoint_callback_best,
        # checkpoint_callback_last,
        lr_callback,
        lr_logger,
    ]

trainer = NotALightningTrainer(
    args = args,
    callbacks = [
        checkpoint_callback_best,
        checkpoint_callback_last,
        lr_callback,
        lr_logger
    ],
    logger=wandb_logger,
)

torch.backends.cudnn.benchmark = True
trainer.fit(
    model,
    train_dataloader,
    evaluators = evaluators
)
