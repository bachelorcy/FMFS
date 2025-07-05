# FMFS
FMFS: Camouflaged Object Detection via Fusion of Multi-scale Frequency and Spatial Domain Features

> Authors:
Yu Chen, Chen Luo, Yong Liu, JiChuan Quan*, Liu Chengzhuo
> 
> If you have any questions about our paper, feel free to contact [Jichuan Quan](qjch_cn@sina.com) 
or [Yu Chen](924715648@qq.com) via E-mail. And if you are using SINet or evaluation toolbox for your research, please cite this paper.
>
>File Structure (PROJECT_STRUCTURE.md)

> Proposed Baseline
The training and testing experiments are conducted using PyTorch with a single GeForce RTX 2080Ti GPU of 11 GB Memory.
>
>Configuring your environment (Prerequisites):
Note that FMFS is only tested on Ubuntu OS with the following environments. It may work on other operating systems as well but we do not guarantee that it will.
>
>Creating a virtual environment in terminal: conda create -n fmfs python=3.10.
>Installing necessary packages: pip install -r requirements.txt. (Under CUDA-11.8 and Pytorch-2.3.1).
>
>Downloading Training and Testing Sets:
>
>★★Downloading NEW training dataset (COD10K-train + CAMO-train) and move it into ./data/Train/, which can be found in this Baidu Pan link : https://pan.baidu.com/s/1DeW8zJ-FeMROysHGZCu2jQ?pwd=esm8


>
>★★Downloading NEW testing dataset (COD10K-test+NC4K+CAMO-test) which can be found in this Baidu Pan link: https://pan.baidu.com/s/1a_ROCUzh3eI0r3M22EVHnA?pwd=1234
> 
> Training and Abolation experiments Configuration:
> 
>★★Backbone: downloading the pre-trained model (pvt_v2_b4.pth + mambavision_small_1k.pth.tar) and move it into ./snapshot/. Baidu Pan link of PVT : https://pan.baidu.com/s/1ecF4bl9c-d1OCo6qxD4QUw?pwd=jyd7 ;
>
>★★Baidu Pan link of mambavision: https://pan.baidu.com/s/1Ba6m6SwZGbu0yOya5fVvQw?pwd=hrzf 
> 
>PvTv2 as backbone:
> 
>Trainer(pvt).py: train FMFS (include abolation experiment: without FRB or without FEM).
>
>★★Baidu Pan link of FEM of Pre-trained model: https://pan.baidu.com/s/1WRnc29NPvScEFWePsmrebQ?pwd=9tf2;
>
>★★Baidu Pan link of FRB of Pre-trained model: https://pan.baidu.com/s/13Fx1BRXvJnaWSTKpsOuBog?pwd=qvgu
>
>★★Trainer(Oct).py: train FMFS without Octave Convolution and NCD. Baidu Pan link of Pre-trained model: https://pan.baidu.com/s/1r_pPRLCtPyoIec5K7-D72g?pwd=zhux
>
>★★Trainer(Abolation).py: the network FMFS only have backbone PvTv2. Baidu Pan link of Pre-trained model: https://pan.baidu.com/s/17RiSk-FUB2yvx3s_SJMXrw?pwd=336s

>Testing Configuration:
>After you download all the pre-trained model and testing data, just run Test.py to generate the final prediction map: replace your trained model directory (--checkpoints) and assign your the save directory of the inferred mask (--Result).
>
>Later, We will try different backbones based FMFS to improve performance and provide more comprehensive comparison.
>

>Results Download
>★★Results of our FMFS can be found in this Baidu Pan link :https://pan.baidu.com/s/1LCbHGqAkgcObXbZScXen7w?pwd=vfv8
