# Point Cloud Quality Enhancement

This repository contains the updated papers about the quality enhancement of point clouds, including three topics:

- **Point cloud completion** 

- **Point cloud up-sampling/super resolution**

- **Point cloud denoising**

# Cite this work
```
@journal{Chen2022,
  title={Survey on deep learning-based point cloud quality enhancement},
  author={Chen, Jianwen and
          Zhao, Lili and
          Ren, Lancao and
          Sun, Zhuoqun and
          Zhang, Xinfeng and 
          Ma, Siwei},
  booktitle={Journal of Image and Graphics},
  year={2023}
}
```

## **1. Point cloud completion**
### **1.1 Voxel-based methods**

[1] VConv-DAE: Deep Volumetric Shape Learning Without Object Labels (ECCV 2016) [[Paper](https://arxiv.org/abs/1604.03755)] [[Code](https://github.com/Not-IITian/VCONV-DAE)]

[2] Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis (CVPR 2017) [[Paper](https://arxiv.org/abs/1612.00101)] [[Code](https://github.com/angeladai/cnncomplete)]

[3] Shape Completion Enabled Robotic Grasping (IROS 2017) [[Paper](https://arxiv.org/abs/1609.08546)] [[Code](https://github.com/CRLab/pc_object_completion_cnn)]

[4] High-Resolution Shape Completion Using Deep Neural Networks for
Global Structure and Local Geometry Inference (ICCV 2017) [[Paper](https://arxiv.org/abs/1709.07599)] 

[5] Learning 3D Shape Completion from Laser Scan Data with Weak Supervision (CVPR 2018) [[Paper](https://arxiv.org/abs/1805.07290)] [[Code](https://github.com/davidstutz/cvpr2018-shape-completion)]

[6] Point-Voxel CNN for Efficient 3D Deep Learning (NeurIPS 2019) [[Paper](https://arxiv.org/abs/1907.03739)] [[Code](https://github.com/mit-han-lab/pvcnn)]

[7] GRNet: Gridding Residual Network for Dense Point Cloud Completion (ECCV 2020) [[Paper](https://arxiv.org/abs/2006.03761)] [[Code](https://github.com/hzxie/GRNet)]

[8] Voxel-based Network for Shape Completion by Leveraging Edge Generation (ICCV 2021) [[Paper](https://ieeexplore.ieee.org/document/9710071)] [[Code](https://github.com/xiaogangw/VE-PCN)]


### **1.2 Point-based methods**
####  **1.2.1 Encoder-decoder based methods**
- #### **Encoder-deocder-common structure**

  [1] PCN: Point Completion Network (3DV 2018) [[Paper](https://arxiv.org/abs/1808.00671)] [[Code](https://github.com/wentaoyuan/pcn)]

  [2] TopNet: Structural Point Cloud Decoder (CVPR 2019) [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)] [[Code](https://github.com/lynetcha/completion3d)]

  [3] Morphing and Sampling Network for Dense Point Cloud Completion (AAAI 2020) [[Paper](https://arxiv.org/abs/1912.00280)] [[Code](https://github.com/Colin97/MSN-Point-Cloud-Completion)]

  [4] Multi-stage point completion network with critical set supervision (Computer Aided Geometric Design 2020) [[Paper](https://www.sciencedirect.com/science/article/pii/S0167839620301126?via%3Dihub)]

  [5] Point Cloud Completion by Skip-attention Network with Hierarchical Folding (CVPR 2020) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wen_Point_Cloud_Completion_by_Skip-Attention_Network_With_Hierarchical_Folding_CVPR_2020_paper.pdf)] 

  [6] ASHF-Net: Adaptive Sampling and Hierarchical Folding Network for Robust Point Cloud Completion (AAAI 2020) [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16478/16285)] 

  [7] Ecg: Edge-aware point cloud completion with graph convolution (IEEE Robotics and Automation Letters 2020) [[Paper](https://ieeexplore.ieee.org/document/9093117)] [[Code](https://github.com/paul007pl/ECG)]

  [8] Learn the 3d object shape completion of point cloud neighborhood information (IEEE Robotics and Automation Letters 2020) [[Paper](http://qikan.cqvip.com/Qikan/Article/Detail?id=7107097755)]

  [9] SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification (ECCV 2020) [[Paper](https://arxiv.org/abs/2008.07358)] [[Code](https://github.com/wangyida/softpool)]

  [10] SAUM: Symmetry-aware upsampling module for consistent point cloud completion (ACCV 2020) [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Son_SAUM_Symmetry-Aware_Upsampling_Module_for_Consistent_Point_Cloud_Completion_ACCV_2020_paper.pdf)] [[Code](https://github.com/countywest/SAUM)]

  [11] FinerPCN: High fidelity point cloud completion network using pointwise convolution (Neurocomputing 2021) [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221010109?via%3Dihub)] [[Code](https://github.com/Colin97/MSN-Point-Cloud-Completion)]

  [12] PCTMA-Net: Point cloud transformer with morphing atlas-based point generation network for dense point cloud completion (IROS 2021) [[Paper](https://ieeexplore.ieee.org/document/9636483)] 

  [13] PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers (ICCV 2021) [[Paper](https://arxiv.org/abs/2108.08839)] [[Code](https://github.com/yuxumin/PoinTr)]

  [14] Cross-Regional Attention Network for Point Cloud Completion (ICPR 2021) [[Paper](https://ieeexplore.ieee.org/document/9413104)] [[Code](https://github.com/paul007pl/ECG)]

  [15] Variational Relational Point Completion Network (CVPR 2021) [[Paper](https://arxiv.org/abs/2104.10154)] [[Code](https://github.com/paul007pl/VRCNet)]

  [16] ASFM-Net: Asymmetrical Siamese Feature Matching Network for Point Completion (ACM MM 2021) [[Paper](https://arxiv.org/abs/2104.09587)] [[Code](https://github.com/Yan-Xia/ASFM-Net)]

  [17] RFNet: Recurrent Forward Network for Dense Point Cloud Completion (ICCV 2021) [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_RFNet_Recurrent_Forward_Network_for_Dense_Point_Cloud_Completion_ICCV_2021_paper.pdf)] 

  [18] Relationship-based point cloud completion (TVCG 2021) [[Paper](https://ieeexplore.ieee.org/document/9528986)]

  [19] Temporal Point Cloud Completion with Pose Disturbance (IEEE Robotics and Automation Letters 2022) [[Paper](https://ieeexplore.ieee.org/document/9695368)]

  [20] Mutual Information Maximization based Similarity Operation for 3D Point Cloud Completion Network (SPL 2022) [[Paper](https://ieeexplore.ieee.org/document/9741313)]

  [21] Learning a Structured Latent Space for Unsupervised Point Cloud Completion (CVPR 2022) [[Paper](https://arxiv.org/abs/2203.15580)]

  [22] Learning Local Displacements for Point Cloud Completion (CVPR 2022) [[Paper](https://arxiv.org/abs/2203.16600)] [[Code](https://github.com/wangyida/disp3d)]

  [23] LAKe-Net: Topology-Aware Point Cloud Completion by Localizing Aligned Keypoints (CVPR 2022) [[Paper](https://arxiv.org/abs/2203.16771)]


- #### **Encoder-Decoder-GAN**

  [1] RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for
Real-Time Point Cloud Shape Completion (CVPR 2019) [[Paper](https://arxiv.org/abs/1904.12304)] [[Code](https://github.com/iSarmad/RL-GAN-Net)]

  [2] Unpaired Point Cloud Completion on Real Scans using Adversarial Training (ICLR 2020) [[Paper](https://arxiv.org/abs/1904.00069)] [[Code](https://github.com/xuelin-chen/pcl2pcl-gan-pub)]

  [3] Cascaded Refinement Network for Point Cloud Completion (CVPR 2020) [[Paper](https://arxiv.org/abs/2004.03327)] [[Code](https://github.com/xiaogangw/cascaded-point-completion)]

  [4] Point cloud completion by learning shape priors (IROS 2020) [[Paper](https://arxiv.org/abs/2008.00394)] [[Code](https://github.com/xiaogangw/point-cloud-completion-shape-prior)]

  [5] PF-Net: Point Fractal Network for 3D Point Cloud Completion (CVPR 2020) [[Paper](https://arxiv.org/abs/2003.00410)] [[Code](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)]

  [6] Dense point cloud completion based on generative adversarial network (TGRS 2021) [[Paper](https://ieeexplore.ieee.org/document/9528913)] [[Code](https://github.com/xuelin-chen/pcl2pcl-gan-pub)]

  [7] Point cloud completion networks based on the generativeadversarial model with self-attention (Journal of China University of Metrology 2021)  [[Paper](http://qikan.cqvip.com/Qikan/Article/Detail?id=7106047629)]

  [8] Multi-feature fusion point cloud completion network (WWW 2021) [[Paper](https://link.springer.com/article/10.1007/s11280-021-00938-8)]

  [9] Towards point cloud completion: Point rank sampling and cross-cascade graph CNN (Neurocomputing 2021) [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221010791)]

  [10] Cascaded Refinement Network for Point Cloud Completion with Self-supervision (TPAMI 2021) [[Paper](https://arxiv.org/abs/2010.08719)] [[Code](https://github.com/xiaogangw/cascaded-point-completion)]

  [11] Cycle4Completion: Unpaired Point Cloud Completion using Cycle Transformation with Missing Region Coding (CVPR 2021) [[Paper](https://yushen-liu.github.io/main/pdf/LiuYS_CVPR21_Cycle4Completion.pdf)] [[Code](https://github.com/diviswen/Cycle4Completion)]

  [12] 3D Point Cloud Shape Completion GAN (Computer Science 2021) [[Paper](http://qikan.cqvip.com/Qikan/Article/Detail?id=7104298091)]

  [13] Multi-scale Transformer based point cloud completion network (Journal of Image and Graphics 2022) [[Paper](http://qikan.cqvip.com/Qikan/Article/Detail?id=7106801908)]

####  **1.2.2 Others**

   [1] A Papier-Mˆach´e Approach to Learning 3D Surface Generation (CVPR 2018) [[Paper](https://arxiv.org/abs/1802.05384)] [[Code](https://github.com/ThibaultGROUEIX/AtlasNet)]

   [2] Detail Preserved Point Cloud Completion via Separated Feature Aggregation (ECCV 2020) [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700511.pdf)] [[Code](https://github.com/XLechter/Detail-Preserved-Point-Cloud-Completion-via-SFA)]

   [3] Skeleton-bridged point completion: From global inference to local adjustment (NIPS 2020) [[Paper](https://arxiv.org/abs/2010.07428)] 

   [4] Vaccine-style-net: Point cloud completion in implicit continuous function space (ACM MM 2020) [[Paper](https://dl.acm.org/doi/epdf/10.1145/3394171.3413648) [[Code](https://github.com/YanWei123/Vaccine-style-net-Point-Cloud-Completion-in-Implicit-Continuous-Function-Space)]

   [5] Unsupervised 3D shape completion through GAN inversion (CVPR 2021) [[Paper](https://arxiv.org/abs/2104.13366v1) [[Code](https://github.com/junzhezhang/shape-inversion)]

   [6] Style-based Point Generator with Adversarial Rendering for Point Cloud Completion (CVPR 2021) [[Paper](https://arxiv.org/abs/2103.02535)] [[Code](https://github.com/microsoft/SpareNet)]

   [7] PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths (CVPR 2021) [[Paper](https://arxiv.org/abs/2012.03408)] [[Code](https://github.com/diviswen/PMP-Net)]

   [8] SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021) [[Paper](https://arxiv.org/abs/2108.04444)] [[Code](https://github.com/AllenXiangX/SnowflakeNet)]

   [9] PMP-Net++: Point Cloud Completion by Transformer-Enhanced Multi-step Point Moving Paths (TPAMI 2022) [[Paper](https://arxiv.org/pdf/2202.09507.pdf)] [[Code](https://github.com/diviswen/PMP-Net)]

   [10] Flow-Based Point Cloud Completion Network with Adversarial Refinement (ICASSP 2022) [[Paper](https://ieeexplore.ieee.org/document/9747024)] 
   
   [11] SeedFormer: patch seeds based point cloud completion with upsample Transformer (ECCV 2022) [[Paper](https://arxiv.org/abs/2207.10315)] [[Code](https://github.com/hrzhou2/seedformer)]


## **2. Point cloud up-sampling/super resolution**

### **2.1 CNN-based methods**

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

### **2.2 GCN-based methods**

[1] Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud (IEEE TVCG 2021) [[Paper](https://arxiv.org/pdf/2102.04317)] [[Code](https://github.com/pleaseconnectwifi/Meta-PU)]

[2] PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks (CVPR 2021) [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf)] [[Code](https://github.com/guochengqian/PU-GCN)]

[3] Deep Magnification-Flexible Upsampling Over 3D Point Clouds (IEEE TIP 2021) [[Paper](https://arxiv.org/pdf/2011.12745)] [[Code](https://github.com/ninaqy/Flexible-PU)]

[4] PU-EVA: An Edge-Vector based Approximation Solution for Flexible-scale Point Cloud Upsampling (CVPR 2021) [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Luo_PU-EVA_An_Edge-Vector_Based_Approximation_Solution_for_Flexible-Scale_Point_Cloud_ICCV_2021_paper.pdf)]

[5] Semantic Point Cloud Upsampling (IEEE TMM 2022) [[Paper](https://ieeexplore.ieee.org/document/9738472)] [[Code](https://github.com/lizhuangzi/SPU)]

[6] PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows (IEEE TVCG 2022) [[Paper](https://ieeexplore.ieee.org/document/9738472)] [[Code](https://github.com/lizhuangzi/SPU)]

[7] Self-supervised arbitrary-scale point clouds upsampling via implicit neural representation (CVPR2022) [[Paper](https://arxiv.org/abs/2204.08196)] [[Code](https://github.com/xnowbzhao/sapcu)]

[8] SPU-Net: self-supervised point cloud upsampling by coarse-to-fine Reconstruction With Self-Projection Optimization (TIP 2022) [[Paper]([https://arxiv.org/abs/2204.08196])] 

[9] Neural points: point cloud representation with neural fields for arbitrary upsampling (CVPR2022) [[Paper]([https://arxiv.org/abs/2204.08196])] [[Code]([https://github.com/xnowbzhao/sapcu])]

[10] BIMS-PU: Bi-Directional and Multi-Scale Point Cloud Upsampling (IRAL 2022) [[Paper]([[https://arxiv.org/abs/2204.08196])]


### **2.3 GAN-based methods**

[1] PU-GAN: A Point Cloud Upsampling Adversarial Network (ICCV 2019) [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_PU-GAN_A_Point_Cloud_Upsampling_Adversarial_Network_ICCV_2019_paper.pdf)] [[Code](https://liruihui.github.io/publication/PU-GAN/)]

[2] Point cloud super-resolution based on geometric constraints (IET Computer Vision 2020) [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.12045)]

[3] PU-Refiner: A Geometry Refiner with Adversarial Learning for Point Cloud Upsampling (ICASSP 2022) [[Paper](https://ieeexplore.ieee.org/abstract/document/9746373)] [[Code](https://github.com/liuhaoyun/PU-Refiner)]

[4] Zero-Shot" point cloud upsampling (ICME2022) [[Paper](https://arxiv.org/abs/2106.13765)] [[Code](https://github.com/ky-zhou/ZSPU)]


## **3. Point cloud denoising**

### **3.1 Encoder-Decoder based methods**

[1] 3D Shape Processing by Convolutional Denoising Autoencoders on Local Patches (WACV 2018)[[Paper](https://www.computer.org/csdl/proceedings-article/wacv/2018/488601b925/12OmNwJgAJQ)]

[2] 3D Point Cloud Denoising via Deep Neural Network based Local Surface Estimation (ICASSP2019) [[Paper](https://arxiv.org/abs/1904.04427)] [[Code](https://github.com/chaojingduan/Neural-Projection)] 

[3] Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning (ICCV 2019) [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hermosilla_Total_Denoising_Unsupervised_Learning_of_3D_Point_Cloud_Cleaning_ICCV_2019_paper.pdf)] [[Code](https://github.com/phermosilla/TotalDenoising)] 

[4] Pointfilter: Point Cloud Filtering via Encoder-Decoder Modeling (IEEE TVCG 2020) [[Paper](https://ieeexplore.ieee.org/document/9207844)] [[Code](http://github.com/dongbo-BUAA-VR/Pointfilter)] 

[5] Differentiable Manifold Reconstruction for Point Cloud Denoising (ACM MM 2020) [[Paper](https://arxiv.org/abs/2007.13551)] [[Code](https://github.com/luost26/DMRDenoise)] 

[6] Reflective Noise Filtering of Large-Scale Point Cloud Using Transformer (Remote Sens 2022) [[Paper](https://www.mdpi.com/2072-4292/14/3/577)]

### **3.2 Others**
[1] PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds (ComputGraphForum 2020) [[Paper](https://arxiv.org/abs/1901.01060v2)] [[Code](https://github.com/mrakotosaon/pointcleannet)] 

[2] Learning Graph-Convolutional Representations for Point Cloud Denoising (ECCV 2020) [[Paper](https://arxiv.org/abs/2007.02578v1)] [[Code](https://github.com/diegovalsesia/GPDNet)] 

[3] Learning Robust Graph-Convolutional Representations for Point Cloud Denoising (IEEE JSTSP 2021) [[Paper](https://ieeexplore.ieee.org/document/9309029)]

[4] Score-Based Point Cloud Denoising (ICCV 2021) [[Paper](https://ieeexplore.ieee.org/document/9711416)] [[Code](https://github.com/luost26/score-denoise)] 

[5] RePCD-Net: Feature-Aware Recurrent Point Cloud Denoising Network (IJCV2022) [[Paper](https://link.springer.com/article/10.1007/s11263-021-01564-7)]

[6] Deep Point Set Resampling via Gradient Fields (TPAMI 2022) [[Paper](https://arxiv.org/abs/2111.02045)]

