# TPRNet
code for "Fusing deep and handcrafted features for intelligent recognition of uptake patterns on thyroid scintigraphy".

## Requirements
- PyTorch 1.0+
- Python 3.x
## Running
train:
```bash
python train.py --config cfgs/tprnet.yaml --gpu 0 --net tprnet --model tpr_model
```
eval:
```bash
python eval.py --config cfgs/tprnet.yaml --gpu 0 --net tprnet --model tpr_model
```
## Datasets
The [data](data/) folder contains a portion of our dataset.