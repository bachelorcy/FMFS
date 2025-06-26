# FMFS
FMFS: Camouflaged Object Detection via Fusion of Multi-scale Frequency and Spatial Domain Features
FMFS-Net/
│   ├── Dataset.py
│   ├── Test.py
│   ├── Train(Abolation).py
│   ├── Trainer(Oct).py
│   ├── Trainer(mambavision).py
│   ├── Trainer(pvt).py
│   ├── requirements.txt
│   ├── logs/
│   │   ├── training_loss.csv
│   ├── val_pred/
│   ├── Results/
│   │   ├── CAMO/
│   │   ├── COD10K/
│   │   ├── NC4K/
│   ├── lib/
│   │   ├── PVTv2.py
│   │   ├── mamba_vision.py
│   │   ├── module.py
│   ├── Network/
│   │   ├── Network_Abolation.py
│   │   ├── Network_MambaVision.py
│   │   ├── Network_PVTv2.py
│   │   ├── Network_PVTv2_NFEM.py
│   │   ├── Network_PVTv2_NFRB.py
│   │   ├── Network_PVTv2_NOct.py
│   ├── data/
│   │   ├── Val/
│   │   │   ├── GT_Object/
│   │   │   ├── Image/
│   │   ├── Train/
│   │   │   ├── GT_Object/
│   │   │   ├── Image/
│   │   ├── Test/
│   │   │   ├── CAMO/
│   │   │   │   ├── GT/
│   │   │   │   ├── Img/
│   │   │   ├── COD10K/
│   │   │   │   ├── GT/
│   │   │   │   ├── Img/
│   │   │   ├── NC4K/
│   │   │   │   ├── GT/
│   │   │   │   ├── Img/
│   ├── utils/
│   │   ├── eval.py
│   │   ├── metrics.py
│   ├── checkpoints/
│   │   ├── Net_epoch_best(ABO).pth
│   │   ├── Net_epoch_best(fem).pth
│   │   ├── Net_epoch_best(frb).pth
│   │   ├── Net_epoch_best(oct).pth
│   │   ├── Net_epoch_best(pvt).pth
│   ├── snapshot/
│   │   ├── mambavision_small_1k.pth.tar
│   │   ├── pvt_v2_b4.pth
