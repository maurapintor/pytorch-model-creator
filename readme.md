# Pretrained PyTorch models

## Train a model

From inside the root directory of the project, run the 
python script from the shell:

```shell script
python train_base_model.py <arguments>
```

For the complete list of options and parameters, run the following 
command: 

```shell script
python train_base_model.py --help
```

#### Train a robust model

If the parameter `--robust` is specified, we can train the model 
with an additional term on the loss, with a technique called 
**dobule backpropagation**. This method computes the loss of the model's 
output, then adds to the original loss the l2 norm of the gradients size 
with respect to input. This term should be computed with the first 
backpropagation, then minimized applying another backprop.

Dobule backpropagation is described [here](https://arxiv.org/pdf/1906.06637.pdf).

### Train the distilled model

For training a distilled model, we need to use a model trained with 
the previous command. Then, we can finally train the model by 
running the command: 

```shell script
python distillation.py <arguments>
```

Again, for the complete list of options and arguments run:

```shell script
python distilled.py --help
```

## Use a pretrained model

TODO