# GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF (ICRA 2023)

This is the official repository of [**GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF**](https://arxiv.org/abs/2210.06575).

For more information, please visit our [**project page**](https://pku-epic.github.io/GraspNeRF/).

## Introduction
![teaser](images/teaser.png)

In this work, we propose a multiview RGB-based 6-DoF grasp detection network, **GraspNeRF**, 
that leverages the **generalizable neural radiance field (NeRF)** to achieve material-agnostic object grasping in clutter. 
Compared to the existing NeRF-based 3-DoF grasp detection methods that rely on densely captured input images and time-consuming per-scene optimization, 
our system can perform zero-shot NeRF construction with **sparse RGB inputs** and reliably detect **6-DoF grasps**, both in **real-time**. 
The proposed framework jointly learns generalizable NeRF and grasp detection in an **end-to-end** manner, optimizing the scene representation construction for the grasping. 
For training data, we generate a large-scale photorealistic **domain-randomized synthetic dataset** of grasping in cluttered tabletop scenes that enables direct transfer to the real world. 
Experiments in synthetic and real-world environments demonstrate that our method significantly outperforms all the baselines in all the experiments.

## Overview
This repository provides:
- PyTorch code and weights of GraspNeRF: Coming soon.
- Multiview 6-DoF Grasping Dataset: Coming soon.


## Citation
If you find our work useful in your research, please consider citing:

```
@article{Dai2023GraspNeRF,
  title={GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF},
  author={Qiyu Dai and Yan Zhu and Yiran Geng and Ciyu Ruan and Jiazhao Zhang and He Wang},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
```

## Contact
If you have any questions, please open a github issue or contact us:

Qiyu Dai: qiyudai@pku.edu.cn, Yan Zhu: zhuyan_@stu.pku.edu.cn, He Wang: hewang@pku.edu.cn