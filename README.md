# WiFi-Person-ReID

<b>Time-Frequency Analysis of Variable-Length WiFi CSI Signals for Person Re-Identification</b>

This repository contains the official python implementation for our paper "Time-Frequency Analysis of Variable-Length WiFi CSI Signals for Person Re-Identification, Chen Mao". 
Our paper are available at [here](https://arxiv.org/abs/2406.01906).

## Introduction

Person re-identification (ReID), as a crucial technology in the field of security, plays an important role in security detection and people counting. Current security and monitoring systems largely rely on visual information, which may infringe on personal privacy and be susceptible to interference from pedestrian appearances and clothing in certain scenarios. Meanwhile, the widespread use of routers offers new possibilities for ReID. 

<img align="center" width="80%" src="https://github.com/Chain-Mao/WiFi-Person-ReID/blob/master/wifi_scene.png">

This letter introduces a method using WiFi Channel State Information (CSI), leveraging the multipath propagation characteristics of WiFi signals as a basis for distinguishing different pedestrian features. We propose a two-stream network structure capable of processing variable-length data, which analyzes the amplitude in the time domain and the phase in the frequency domain of WiFi signals, fuses time-frequency information through continuous lateral connections, and employs advanced objective functions for representation and metric learning. Tested on a dataset collected in the real world, our method achieves 93.68% mAP and 98.13% Rank-1.

<img align="center" width="80%" src="https://github.com/Chain-Mao/WiFi-Person-ReID/blob/master/wifi_arch.png">

## Train

#### Stage1

The purpose of the first stage training is to generate prompts that describe the image.

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/stage1.png">

After downloading the SF-XL dataset, you can start the stage1 training as such

`$ python3 train_clip_stage1.py --train_set_folder path/to/processed/train --val_set_folder path/to/sf_xl/processed/val --test_set_folder path/to/sf_xl/processed/test --backbone CLIP-RN50 --groups_num 1`

#### Stage2

The purpose of the second stage training is to use prompts to assist the image model to complete the clustering.

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/stage2.png">

After generating the prompts through stage1 training, you can start the stage2 training as such

`$ python3 train_clip_stage2.py --train_set_folder path/to/processed/train --val_set_folder path/to/processed/val --test_set_folder path/to/processed/test --backbone CLIP-RN50 --fc_output_dim 1024 --prompt_learners path/to/logs/default/stage1/VIT16/last_prompt_learners.pth`

To change the backbone and the output descriptors dimensionality simply run 

`$ python3 train.py --backbone CLIP-ViT-B-16 --fc_output_dim 512`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

## Test

You can test a trained model as such

`$ python3 eval.py --backbone CLIP-RN50 --resume_model path/to/best_model.pth --test_set_folder path/to/processed/test`

<img align="center" width="80%" src="https://github.com/Chain-Mao/ProGEO/blob/main/visual.png">

## Issues
If you have any questions regarding our code or model, feel free to open an issue or send an email to maochen981203@gmail.com

## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [fast-reid](https://github.com/JDAI-CV/fast-reid)

## Cite
Here is the bibtex to cite our arxiv paper, the Springer version will be cited after official publication.
```
@ARTICLE{2024arXiv240601906M,
       author = {{Mao}, Chen and {Hu}, Jingqi},
        title = "{ProGEO: Generating Prompts through Image-Text Contrastive Learning for Visual Geo-localization}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Information Retrieval},
         year = 2024,
        month = jun,
          eid = {arXiv:2406.01906},
        pages = {arXiv:2406.01906},
          doi = {10.48550/arXiv.2406.01906},
archivePrefix = {arXiv},
       eprint = {2406.01906},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240601906M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
