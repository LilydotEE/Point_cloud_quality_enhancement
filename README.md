# Point cloud quality enhancement

The updated papers about the quality enhancement of point clouds, including three topics:

- point cloud completion 

- point cloud up-sampling/super resolution

- point cloud denoising



## **Point cloud completion**

- ### Voxel-based methods

[1] VConv-DAE: Deep Volumetric Shape Learning Without Object Labels (ECCV 2016) [[Paper](https://arxiv.org/abs/1604.03755)] [[Code](https://github.com/Not-IITian/VCONV-DAE)]

[2] Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis (CVPR 2017) [[Paper](https://arxiv.org/abs/1612.00101)] [[Code](https://github.com/angeladai/cnncomplete)]

[3] Shape Completion Enabled Robotic Grasping (IROS 2017) [[Paper](https://arxiv.org/abs/1609.08546)] [[Code](https://github.com/CRLab/pc_object_completion_cnn)]

[4] High-Resolution Shape Completion Using Deep Neural Networks for
Global Structure and Local Geometry Inference (ICCV 2017) [[Paper](https://arxiv.org/abs/1709.07599)] 

[5] Learning 3D Shape Completion from Laser Scan Data with Weak Supervision (CVPR 2018) [[Paper](https://arxiv.org/abs/1805.07290)] [[Code](https://github.com/davidstutz/cvpr2018-shape-completion)]

[6] Point-Voxel CNN for Efficient 3D Deep Learning (NeurIPS 2019) [[Paper](https://arxiv.org/abs/1907.03739)] [[Code](https://github.com/mit-han-lab/pvcnn)]

- ### Point-based methods

[1] A Papier-Mˆach´e Approach to Learning 3D Surface Generation (CVPR 2018) [[Paper](https://arxiv.org/abs/1802.05384)] [[Code](https://github.com/ThibaultGROUEIX/AtlasNet)]

[2] PCN: Point Completion Network (3DV 2018) [[Paper](https://arxiv.org/abs/1808.00671)] [[Code](https://github.com/wentaoyuan/pcn)]

[3] RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for
Real-Time Point Cloud Shape Completion (CVPR 2019) [[Paper](https://arxiv.org/abs/1904.12304)] [[Code](https://github.com/iSarmad/RL-GAN-Net)]

[4] TopNet: Structural Point Cloud Decoder (CVPR 2019) [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)] [[Code](https://github.com/lynetcha/completion3d)]

[5] Morphing and Sampling Network for Dense Point Cloud Completion (AAAI 2020) [[Paper](https://arxiv.org/abs/1912.00280)] [[Code](https://github.com/Colin97/MSN-Point-Cloud-Completion)]

[6] Unpaired Point Cloud Completion on Real Scans using Adversarial Training (ICLR 2020) [[Paper](https://arxiv.org/abs/1904.00069)] [[Code](https://github.com/xuelin-chen/pcl2pcl-gan-pub)]

[7] Point Cloud Completion by Skip-attention Network with Hierarchical Folding (CVPR 2020) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wen_Point_Cloud_Completion_by_Skip-Attention_Network_With_Hierarchical_Folding_CVPR_2020_paper.pdf)] 

[8] PF-Net: Point Fractal Network for 3D Point Cloud Completion (CVPR 2020) [[Paper](https://arxiv.org/abs/2003.00410)] [[Code](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)]

## **Point cloud up-sampling/super resolution**

- ### CNN-based methods

[1] PU-Net: Point Cloud Upsampling Network (CVPR 2018) [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.pdf)] [[Code](https://github.com/yulequan/PU-Net)]

[2] EC-Net: an Edge-aware Point set Consolidation Network (ECCV 2018) [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Lequan_Yu_EC-Net_an_Edge-aware_ECCV_2018_paper.pdf)] [[Code](https://yulequan.github.io/ec-net/index.html)]

[3] Patch-based Progressive 3D Point Set Upsampling (CVPR 2019) [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yifan_Patch-Based_Progressive_3D_Point_Set_Upsampling_CVPR_2019_paper.pdf)] [[Code](https://github.com/yifita/3pu)]

[4] PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling (ECCV 2020) [[Paper](https://arxiv.org/pdf/2002.10277)] [[Code](https://github.com/ninaqy/PUGeo)]

[5] Point Cloud Upsampling via Disentangled Refinement (CVPR 2021) [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Point_Cloud_Upsampling_via_Disentangled_Refinement_CVPR_2021_paper.pdf)] [[Code](https://github.com/liruihui/Dis-PU)]

[6] Sequential Point Cloud Upsampling by Exploiting Multi-Scale Temporal Dependency (IEEE TCSVT 2021) [[Paper](https://ieeexplore.ieee.org/abstract/document/9512063)] 

[7] SSPU-Net: Self-Supervised Point Cloud Upsampling via Differentiable Rendering (ACM MM 2021) [[Paper](https://arxiv.org/pdf/2108.00454)] [[Code](https://github.com/fpthink/SSPU-Net)]

[8] Progressive Point Cloud Upsampling via Differentiable Rendering (IEEE TCSVT 2021) [[Paper](https://ieeexplore.ieee.org/abstract/document/9496619)] 

[9] Density-imbalance-eased LiDAR Point Cloud Upsampling via Feature Consistency Learning (IEEE TIV 2022) [[Paper](https://ieeexplore.ieee.org/abstract/document/9743721)]

[10] VPU: A Video-based Point Cloud Upsampling Framework (IEEE TIP 2022) [[Paper](https://ieeexplore.ieee.org/document/9759233)]

- ### GCN-based methods

[1] Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud (IEEE TVCG 2021) [[Paper](https://arxiv.org/pdf/2102.04317)] [[Code](https://github.com/pleaseconnectwifi/Meta-PU)]

[2] PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks (CVPR 2021) [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf)] [[Code](https://github.com/guochengqian/PU-GCN)]

[3] Deep Magnification-Flexible Upsampling Over 3D Point Clouds (IEEE TIP 2021) [[Paper](https://arxiv.org/pdf/2011.12745)] [[Code](https://github.com/ninaqy/Flexible-PU)]

[4] PU-EVA: An Edge-Vector based Approximation Solution for Flexible-scale Point Cloud Upsampling (CVPR 2021) [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Luo_PU-EVA_An_Edge-Vector_Based_Approximation_Solution_for_Flexible-Scale_Point_Cloud_ICCV_2021_paper.pdf)]

[5] Semantic Point Cloud Upsampling (IEEE TMM 2022) [[Paper](https://ieeexplore.ieee.org/document/9738472)] [[Code](https://github.com/lizhuangzi/SPU)]

[6] PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows (IEEE TVCG 2022) [[Paper](https://ieeexplore.ieee.org/document/9738472)] [[Code](https://github.com/lizhuangzi/SPU)]

- ### GAN-based methods

[1] PU-GAN: A Point Cloud Upsampling Adversarial Network (ICCV 2019) [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_PU-GAN_A_Point_Cloud_Upsampling_Adversarial_Network_ICCV_2019_paper.pdf)] [[Code](https://liruihui.github.io/publication/PU-GAN/)]

[2] Point cloud super-resolution based on geometric constraints (IET Computer Vision) [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.12045)]

[3] PU-Refiner: A Geometry Refiner with Adversarial Learning for Point Cloud Upsampling (ICASSP 2022) [[Paper](https://ieeexplore.ieee.org/abstract/document/9746373)] [[Code](https://github.com/liuhaoyun/PU-Refiner)]


## **Point cloud denoising**

### Encoder-Decoder based methods

[1] 3D Shape Processing by Convolutional Denoising Autoencoders on Local Patches (WACV 2018)[[Paper](https://www.computer.org/csdl/proceedings-article/wacv/2018/488601b925/12OmNwJgAJQ)]

[2] 3D Point Cloud Denoising via Deep Neural Network based Local Surface Estimation (ICASSP2019) [[Paper](https://arxiv.org/abs/1904.04427)] [[Code](https://github.com/chaojingduan/Neural-Projection)] 

[3] Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning (ICCV 2019) [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hermosilla_Total_Denoising_Unsupervised_Learning_of_3D_Point_Cloud_Cleaning_ICCV_2019_paper.pdf)] [[Code](https://github.com/phermosilla/TotalDenoising)] 

[4] Pointfilter: Point Cloud Filtering via Encoder-Decoder Modeling (IEEE TVCG 2020) [[Paper](https://ieeexplore.ieee.org/document/9207844)] [[Code](http://github.com/dongbo-BUAA-VR/Pointfilter)] 

[5] Differentiable Manifold Reconstruction for Point Cloud Denoising (ACM MM 2020) [[Paper](https://arxiv.org/abs/2007.13551)] [[Code](https://github.com/luost26/DMRDenoise)] 

[6] Reflective Noise Filtering of Large-Scale Point Cloud Using Transformer (Remote Sens 2022) [[Paper](https://www.mdpi.com/2072-4292/14/3/577)]

### Others
[1] PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds (ComputGraphForum 2020) [[Paper](https://arxiv.org/abs/1901.01060v2)] [[Code](https://github.com/mrakotosaon/pointcleannet)] 

[2] Learning Graph-Convolutional Representations for Point Cloud Denoising (ECCV 2020) [[Paper](https://arxiv.org/abs/2007.02578v1)] [[Code](https://github.com/diegovalsesia/GPDNet)] 

[3] Learning Robust Graph-Convolutional Representations for Point Cloud Denoising (IEEE JSTSP) [[Paper](https://ieeexplore.ieee.org/document/9309029)]

[4] Score-Based Point Cloud Denoising (ICCV 2021) [[Paper](https://ieeexplore.ieee.org/document/9711416)] [[Code](https://github.com/luost26/score-denoise)] 

[5] RePCD-Net: Feature-Aware Recurrent Point Cloud Denoising Network (IJCV2022) [[Paper](https://link.springer.com/article/10.1007/s11263-021-01564-7)]

[6] Deep Point Set Resampling via Gradient Fields (TPAMI 2022) [[Paper](https://arxiv.org/abs/2111.02045)]

