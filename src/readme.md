# Pretrained PyTorch models

## Train a model

From inside the directory `secml-lib/ai_sec_evasion/pretrained`, run the 
python script from the shell:

```shell script
python train_base_model.py <arguments>
```

```shell script
usage: train_base_model.py [-h] [--epochs EPOCHS] [--dataset {mnist,cifar10}]
                           [--nb_train NB_TRAIN] [--nb_test NB_TEST]
                           [--include_list INCLUDE_LIST]
                           [--batch_size BATCH_SIZE] [--lr LR]
                           [--scheduler_steps SCHEDULER_STEPS]
                           [--scheduler_gamma SCHEDULER_GAMMA]
                           [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                           [--use_cuda] [--num_workers NUM_WORKERS]
                           [--output_file OUTPUT_FILE]

Train base model.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs for training (default: 10).
  --dataset {mnist,cifar10}
                        Dataset to use for training (default: mnist).
  --nb_train NB_TRAIN   Number of samples to use for training (default:
                        10000).
  --nb_test NB_TEST     Number of samples to use for testing (default: 2000)
  --include_list INCLUDE_LIST
                        Classes to include in the training process, separated
                        by commas (default: all).
  --batch_size BATCH_SIZE
                        Batch size to use for training (default: 128).
  --lr LR               Learning rate to use for training (default: 0.01).
  --scheduler_steps SCHEDULER_STEPS
                        Learning rate scheduler steps, separated by commas
                        (default: None)
  --scheduler_gamma SCHEDULER_GAMMA
                        Learning rate decay to apply at the milestones defined
                        in scheduler_steps (default: 0).
  --weight_decay WEIGHT_DECAY
                        Weight decay to use as regularization (default: 0).
  --momentum MOMENTUM   Momentum to use during training (default: 0).
  --use_cuda            Use cuda if available (default: True).
  --num_workers NUM_WORKERS
                        Number of additional workers to spawn for the training
                        process (default: 0).
  --output_file OUTPUT_FILE
                        Name of the output file. The file will be stored in
                        the directory `pretrained_models`
```

### Train the distilled model



## Use a pretrained model

