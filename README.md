# Virtual MOLLI Target: Generative Adversarial Networks for Enhanced Motion Correction in Myocardial T1 Mapping

This repository is the work of "_Virtual MOLLI Target: Generative Adversarial Networks for Enhanced Motion Correction in Myocardial T1 Mapping_" based on **pytorch** implementation. You could click the link to access the [paper](xxx).


## MedGAN Architecture

The architecture refers to this [paper](https://www.sciencedirect.com/science/article/pii/S0895611119300990).

<div  align="center">  
 <img src="https://github.com/htylab/VMT/blob/main/doc/MedGAN.png"
     align=center/>
</div>


## Data preprocess

Please prepare the npz file in the following format:

1. CINE format

    'image': the image of cine

    'mask': include LV, RV and LVW

2. VMT format

    'im': the images of MOLLI
    
    'invtime': the sequence of inversion time


## Training

```
>>python train.py --training_model=VMT --batch_size=1 --epoch=100 --save_frq=10 --workers=0 --pretrain=pretrain_weights/CINE.pt --data_root=data/GANdata/ --train_txt=data/train.txt --valid_txt=data/valid.txt

--training_model: Select training CINE or VMT models (default: VMT)
--batch_size: Number of samples processed before model update (default: 1)
--epoch: Training times for the training dataset (default: 100)
--save_frq: How often to store the model weights (default: 10)
--workers: Multi-process data loading (default: 0)
--pretrain(optional): The path of pretain weights
--data_root: The path of datasets
--train_txt: The path of training list
--valid_txt: The path of validation list
```


## Produce VMT

```
python produce_result.py --workers=0 --weights_path=pretrain_weights/VMT.pt --data_root=data/GANdata/ --valid_txt=data/valid.txt

--workers: Multi-process data loading (default: 0)
--weights_path: The path of model weights
--data_root: The path of datasets
--valid_txt: The path of validation list
--save_path: The path of saving the VMT result
```


## Registration

We provide the motion correction method by using synthetic images in this [paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.23153).

```
python registration.py --data_path=results/ --iteration=4 --save_path=results_reg/

--data_path: The path of the VMT result
--iteration: Number of times to repeat registration (default: 4)
--save_path: The path of saving the result
```


## Fitting Quality Index (FQI)

We provide a script to calculate FQI for users to calculate the FQI value after registration.

```
python calculate_FQI.py --data_path=results_reg/

--data_path: The path of the registration result
```


## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```

```