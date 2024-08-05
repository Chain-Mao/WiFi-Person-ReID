# WiFi-Person-ReID

<b>Time-Frequency Analysis of Variable-Length WiFi CSI Signals for Person Re-Identification</b>

This repository contains the official python implementation for our paper "Time-Frequency Analysis of Variable-Length WiFi CSI Signals for Person Re-Identification, Chen Mao". 

Our paper was accepted by IEEE Signal Processing Letters, the paper is available at [here](https://arxiv.org/abs/2407.09045).

<div align="center"><img src="wifi_scene.png"  width="60%"/></div>

## Introduction

Person re-identification (ReID), as a crucial technology in the field of security, plays an important role in security detection and people counting. Current security and monitoring systems largely rely on visual information, which may infringe on personal privacy and be susceptible to interference from pedestrian appearances and clothing in certain scenarios. Meanwhile, the widespread use of routers offers new possibilities for ReID. 

<div align="center"><img src="wifi_arch.png"  width="80%"/></div>
<div align="center"><img src="csi_uniform.png"  width="60%"/></div>

This letter introduces a method using WiFi Channel State Information (CSI), leveraging the multipath propagation characteristics of WiFi signals as a basis for distinguishing different pedestrian features. We propose a two-stream network structure capable of processing variable-length data, which analyzes the amplitude in the time domain and the phase in the frequency domain of WiFi signals, fuses time-frequency information through continuous lateral connections, and employs advanced objective functions for representation and metric learning. Tested on a dataset collected in the real world, our method achieves 93.68% mAP and 98.13% Rank-1.

## Cite
Here is the bibtex to cite our arxiv paper, the Springer version will be cited after official publication.
```
@misc{mao2024timefrequencyanalysisvariablelengthwifi,
      title={Time-Frequency Analysis of Variable-Length WiFi CSI Signals for Person Re-Identification}, 
      author={Chen Mao and Chong Tan and Jingqi Hu and Min Zheng},
      year={2024},
      eprint={2407.09045},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.09045}, 
}
```

Parts of this repo are inspired by [fast-reid](https://github.com/JDAI-CV/fast-reid).
