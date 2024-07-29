# SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning

### [Paper](https://arxiv.org/abs/2407.08148)

## Abstract
We propose a novel unsupervised cross-modal homography estimation framework based on intra-modal Self-supervised learning, Correlation, and consistent feature map Projection, namely SCPNet. The concept of intra-modal self-supervised learning is first presented to facilitate the unsupervised cross-modal homography estimation. The correlation-based homography estimation network and the consistent feature map projection are combined to form the learnable architecture of SCPNet, boosting the unsupervised learning framework. SCPNet is the first to achieve effective unsupervised homography estimation on the satellite-map image pair cross-modal dataset, GoogleMap, under [-32,+32] offset on a 128 × 128 image, leading the supervised approach MHN by 14.0% of mean average corner error (MACE). We further conduct extensive experiments on several cross-modal/spectral and manually-made inconsistent datasets, on which SCPNet achieves the state-of-the-art (SOTA) performance among unsupervised approaches, and owns 49.0%, 25.2%, 36.4%, and 10.7% lower MACEs than the supervised approach MHN.


## Quick Start

Evaluation on the GoogleMap dataset:
```Shell
python test_SCPNet.py --model_dir ./ckpt/ --model_name ggmap.pth --dataset ggmap
```

Training on the GoogleMap dataset:
```Shell
python train_SCPNet.py --model_dir ./result/exp_ggmap --dataset ggmap
```


## License
This project is released under the Apache 2.0 license.


## Contact
If you have any other problems, feel free to post questions in the issues section or contact Runmin Zhang (runmin_zhang@zju.edu.cn) and Si-Yuan Cao (cao_siyuan@zju.edu.cn).


## Acknowledgement
This work is mainly based on [RAFT](https://github.com/princeton-vl/RAFT) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for the contribution.

## Citation

If you find this project helpful, please consider citing the following paper:
```bibtex
@article{zhang2024scpnet,
  title={SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning},
  author={Zhang, Runmin and Ma, Jun and Cao, Si-Yuan and Luo, Lun and Yu, Beinan and Chen, Shu-Jie and Li, Junwei and Shen, Hui-Liang},
  journal={arXiv preprint arXiv:2407.08148},
  year={2024}
}
```