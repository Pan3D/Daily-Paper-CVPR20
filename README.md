# Daily-Paper-CVPR20-in-3D
[Conference Home](<http://cvpr2020.thecvf.com/>) Seattle, Washington 

Get more information, please subscribe the account "**3D Daily**" in Wechat.

![3D Daily](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/3D_Daily.jpg)

Includes 3D [Classification/Segmentation](#CS), [Detection](#Detection), [Generation/Reconstruction](#GenRecons), [3D Face](#3DFace), 
[Pose Estimation](#PoseEstimation), [Matching](#Matching), [Keypoints](#Keypoints), [Layout](#Layout), [Depth](#Depth), [Surfaces](#Surfaces), [Texture](#Texture), [Graph](#Graph), [Dataset](#Dataset), et al.

More details about the papers please see the subject documents for papers' abstraction and pipeline.

-[Classification/Segmentation](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation.md)

-[Detection](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection.md)

-[Generation/Reconstruction](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction.md)

-[Pose Estimation](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation.md)


-[Keypoints, Layout, Depth, Surface](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface.md)

-[Matching, Tecture, Datasets](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets.md)

-[3D Face, Graph, Others](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others.md)

1. The first link "PR20" after each title is the paper's address from the arXiv websites.
2. The link "TF/Torch" means the code based on which platform. "Code" means the code has not released yet.
3. The link "Account" introduces the paper in Chinese from the account in Wechat.
4. The link like "Graph" means this paper also in the "Graph" subject. "C/S" means "Classification/Segmentation", et al.

------

### CS

1. PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation [PR20](https://arxiv.org/pdf/2004.01658.pdf) 
2. PointGLR: Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds [PR20](https://arxiv.org/pdf/2003.12971.pdf) [Torch](https://github.com/raoyongming/PointGLR )
3. 3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation [PR20](https://arxiv.org/pdf/2003.13867.pdf) [Project](https://www.vision.rwth-aachen.de/publication/00199/) [Video](https://youtu.be/ifL8yTbRFDk) 
4. PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation [PR20](https://arxiv.org/pdf/2003.14032.pdf) [Torch](https://github.com/edwardzhou130/PolarSeg) 
5. DualConvMeshNet: Joint Geodesic and Euclidean Convolutions on 3D Meshes [PR20](https://arxiv.org/pdf/2004.01002.pdf) [Torch](https://github.com/VisualComputingInstitute/dcm-net) 
6. Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image [PR20](https://arxiv.org/pdf/2004.01176.pdf) [Torch](https://github.com/paschalidoud/hierarchical_primitives) 
7. Multi-Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds [PR20](https://arxiv.org/pdf/2003.13035.pdf) 
8. Learning to Segment 3D Point Clouds in 2D Image Space [PR20](<https://arxiv.org/pdf/2003.05593.pdf>) [TF](<https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space>)
9. xMUDA： Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [PR20](https://arxiv.org/abs/1911.12676) [Code](<https://github.com/valeoai/xmuda>) 
10. FPConv: Learning Local Flattening for Point Convolution [PR20](<https://arxiv.org/pdf/2002.10701.pdf>) [Torch](<https://github.com/lyqun/FPConv>) 
11. OccuSeg: Occupancy-aware 3D Instance Segmentation [PR20](<https://arxiv.org/pdf/2003.06537.pdf>) [Graph](#Graph) 
12. GridGCN: Grid-GCN for Fast and Scalable Point Cloud Learning [PR20](<https://arxiv.org/pdf/1912.02984.pdf>) [Account](<https://leijiezhang001.github.io/paper-reading-Grid-GCN-for-Fast-and-Scalable-Point-Cloud-Learning/>) [Graph](#Graph) 
13. PointAugment: An Auto-Augmentation Framework for Point Cloud Classification [PR20](<https://arxiv.org/pdf/2002.10876.pdf>) [Code](<https://github.com/liruihui/PointAugment/>) 
14. PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling [PR20](<https://arxiv.org/pdf/2003.00492.pdf>)  [TF](<https://github.com/yanx27/PointASNL>) 
15. RandLANet: Efficient Semantic Segmentation of Large-Scale Point Clouds [PR20](<https://arxiv.org/pdf/1911.11236.pdf>) [TF](<https://github.com/QingyongHu/RandLA-Net>) [Account](<https://zhuanlan.zhihu.com/p/105433460>) 
16. Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels [PR20](<https://arxiv.org/pdf/2004.04091.pdf>) [Code](<https://github.com/alex-xun-xu/WeakSupPointCloudSeg>) 

------

### Detection

1. FocalMix: Semi-Supervised Learning for 3D Medical Image Detection [PR20](https://arxiv.org/pdf/2003.09108.pdf) 
2. LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention [PR20](https://arxiv.org/pdf/2004.01389.pdf) [Code](https://github.com/yinjunbo/3DVID) 
3. Learning to Detect 3D Objects and Predict their 3D Shapes [PR20](https://arxiv.org/pdf/2004.01170.pdf) 
4. PointGNN: Graph Neural Network for 3D Object Detection in a Point Cloud [PR20](<https://arxiv.org/pdf/2003.01251.pdf>)  [TF](<https://github.com/WeijingShi/Point-GNN>) [Graph](#Graph) 
5. MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships [PR20](<https://arxiv.org/pdf/2003.00504.pdf>) 
6. MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps [PR20](<https://arxiv.org/pdf/2003.06754.pdf>) [Code](<https://github.com/pxiangwu/MotionNet>) 
7. HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection [PR20](<https://arxiv.org/pdf/2003.00186.pdf>) [Account](https://mp.weixin.qq.com/s/prSkJIlVLdINBvNI8eUgvA) 
8. Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation [PR20](<https://arxiv.org/pdf/2004.03572.pdf>) [Torch](<https://github.com/zju3dv/disprcnn>) 
9. End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection [PR20](<https://arxiv.org/pdf/2004.03080.pdf>) [Code](<https://github.com/mileyan/pseudo-LiDAR_e2e>) 
10. UCNet: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders [PR20 Oral](https://arxiv.org/pdf/2004.05763.pdf) [Code](https://github.com/JingZhang617/UCNet)
11. MLCVNet: Multi-Level Context VoteNet for 3D Object Detection [PR20](https://arxiv.org/pdf/2004.05679.pdf) [Code](https://github.com/NUAAXQ/MLCVNet)
12. Probabilistic Orientated Object Detection in Automotive Radar [PR20](https://arxiv.org/pdf/2004.05310.pdf) 

------

### GenRecons

1. PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization [PR20](https://arxiv.org/pdf/2004.00452.pdf) [Project](https://shunsukesaito.github.io/PIFuHD/) 
2. Local Implicit Grid Representations for 3D Scenes [PR20](https://arxiv.org/pdf/2003.08981.pdf) [Video](https://www.youtube.com/watch?v=XCyl1-vxfII&feature=youtu.be) 
3. DeepCap: Monocular Human Performance Capture Using Weak Supervision [PR20](<https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/>) 
4. 3DSketchAwareSSC: 3D Sketch-aware Semantic Scene Completion via Semi-supervised Structure Prior [PR20](<https://arxiv.org/pdf/2003.14052.pdf>) [Project](<https://charlescxk.github.io/>) [Code](<https://github.com/charlesCXK/3D-SketchAware-SSC>) 
5. PQNet: A Generative Part Seq2Seq Network for 3D Shapes [PR20](<https://arxiv.org/pdf/1911.10949.pdf>) [Torch](<https://github.com/ChrisWu1997/PQ-NET>) [Account](<https://mp.weixin.qq.com/s/SMDzPAJCpgjELsKHeQvW-A>) 
6. Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion [PR20](<https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf>)  [Code](<https://virtualhumans.mpi-inf.mpg.de/ifnets/>) 
7. InPerfectShape: Certifiably Optimal 3D Shape Reconstruction from 2D Landmarks [PR20](https://arxiv.org/pdf/1911.11924.pdf)
8. PFNet: Point Fractal Network for 3D Point Cloud Completion [PR20](<https://arxiv.org/pdf/2003.00410.pdf>) [Torch](<https://github.com/zztianzz/PF-Net-Point-Fractal-Network>) [Account](<https://mp.weixin.qq.com/s/0b1FeYv6DMj-rLysSi6lYQ>)
9. Cascaded Refinement Network for Point Cloud Completion [PR20](<https://arxiv.org/pdf/2004.03327.pdf>) [Code](<https://github.com/xiaogangw/cascaded-point-completion>)
10. Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions [PR20](<https://arxiv.org/pdf/2004.03967.pdf>) [Video](<https://www.youtube.com/watch?v=8D3HjYf6cYw&feature=youtu.be>) 
11. ARCH: Animatable Reconstruction of Clothed Humans [PR20](https://arxiv.org/pdf/2004.04572.pdf) 

------

### PoseEstimation

1. Towards Better Generalization: Joint Depth-Pose Learning without PoseNet [PR20](https://arxiv.org/pdf/2004.01314.pdf) [Torch](https://github.com/B1ueber2y/TrianFlow) [Depth](#Depth)
2. HandVoxNet: Deep Voxel-Based Network for 3D Hand Shape and Pose Estimation from a Single Depth Map [PR20](https://arxiv.org/pdf/2004.01588.pdf) 
3. ONP: Optical Non-Line-of-Sight Physics-based 3D Human Pose Estimation [PR20](https://arxiv.org/pdf/2003.14414.pdf) [Project](https://marikoisogawa.github.io/project/nlos_pose) 
4. HOPENet: A Graph-based Model for Hand-Object Pose Estimation [PR20](https://arxiv.org/pdf/2004.00060.pdf) [Project](http://vision.sice.indiana.edu/projects/hopenet) [Torch](https://github.com/bardiadoosti/HOPE/) [Account](https://mp.weixin.qq.com/s/SedoU-W2wUNIqqmzKArf1A) [Graph](#Graph)
5. Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation [PR20](https://arxiv.org/pdf/2004.00329.pdf ) [Torch](https://github.com/fabbrimatteo/LoCO) 
6. Estimating 6D Pose of Objects with Symmetries [PR20](https://arxiv.org/pdf/2004.00605.pdf) [Project](http://cmp.felk.cvut.cz/epos/ ) 
7. BodiesatRest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data [PR20](https://arxiv.org/pdf/2004.01166.pdf) 
8. G2LNet: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features [PR20](https://arxiv.org/pdf/2003.11089.pdf) [Torch](https://github.com/DC1991/G2L_Net) 
9. Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS [PR20](<https://arxiv.org/pdf/2003.03972.pdf>)
10. Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild [PR20 Oral](https://arxiv.org/pdf/2004.01946.pdf) [Project](https://www.arielai.com/mesh_hands/) 
11. Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation [PR20](https://arxiv.org/pdf/2004.02186.pdf) 
12. Self-Supervised 3D Human Pose Estimation via Part Guided Novel Image Synthesis [PR20 Oral](https://arxiv.org/pdf/2004.04400.pdf) [Project](http://val.cds.iisc.ac.in/pgp-human/) 
13. MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion [PR20](https://arxiv.org/pdf/2004.04336.pdf) 
14. Weakly-Supervised 3D Human Pose Learning via Multi-view Images in the Wild  [PR20](https://arxiv.org/pdf/2003.07581.pdf) 
15. Video Inference for Human Body Pose and Shape Estimation [PR20](<https://arxiv.org/pdf/1912.05656.pdf>) [Torch](<https://github.com/mkocabas/VIBE>) 

------

### 3DFace

1. THF3D: Towards High-Fidelity 3D Face Reconstruction from In-the-Wild Images Using Graph Convolutional Networks [PR20](<https://arxiv.org/pdf/2003.05653.pdf>) [Graph](#Graph)
2. AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild" [Project](https://github.com/lattas/AvatarMe) 

------
### Matching

1. RPMNet: Robust Point Matching using Learned Features  [PR20](https://arxiv.org/pdf/2003.13479.pdf)  [Code](https://github.com/yewzijian/RPMNet)  
2. Learning Multiview 3D Point Cloud Registration [PR20](<https://arxiv.org/pdf/2001.05119.pdf>) 
3. D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features [PR20](<https://arxiv.org/pdf/2003.03164.pdf>) [TF](<https://github.com/XuyangBai/D3Feat>) 
4. End-to-End Learning Local Multi-view Descriptors for 3D Point Clouds [PR20](<https://arxiv.org/pdf/2003.05855.pdf>) 

------

### Keypoints

1. PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation [PR20](https://arxiv.org/abs/1911.04231)  [Torch](<https://github.com/ethnhe/PVN3D>) [Video](<https://www.bilibili.com/video/av89408773/>) 
2. KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations [PR20](<https://arxiv.org/pdf/2002.12687.pdf>)

------

### Layout

1. Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction  for Indoor Scenes from a Single Image [PR20](<https://arxiv.org/pdf/2002.12212.pdf>) 
2. IntelligentHome3D: Automatic 3D-House Design from Linguistic Descriptions Only [PR20](<https://arxiv.org/pdf/2003.00397.pdf>) 

------

### Depth

1. Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera [PR20](https://arxiv.org/pdf/2004.01294.pdf) 
2. Towards Better Generalization: Joint Depth-Pose Learning without PoseNet [PR20](https://arxiv.org/pdf/2004.01314.pdf) [Torch](https://github.com/B1ueber2y/TrianFlow) [Pose Estimation]
3. Depth Sensing Beyond LiDAR Range [PR20](<https://arxiv.org/pdf/2004.03048.pdf>)

------

### Surface

1. Articulation-aware Canonical Surface Mapping [PR20](https://arxiv.org/pdf/2004.00614.pdf) [Torch](https://github.com/nileshkulkarni/acsm/) [Project](https://nileshkulkarni.github.io/acsm/) 
2. Deep 3D Capture: Geometry and Reflectance from Sparse Multi-View Images [PR20](https://arxiv.org/pdf/2003.12642.pdf) 
3. Where Does It End? -- Reasoning About Hidden Surfaces by Object Intersection Constraints [PR20](https://arxiv.org/pdf/2004.04630.pdf) 

------

### Texture

1. Learning to Transfer Texture from Clothing Images to 3D Humans [PR20](<https://virtualhumans.mpi-inf.mpg.de/papers/mir20pix2surf/mir20pix2surf.pdf>) [Code](<https://virtualhumans.mpi-inf.mpg.de/pix2surf/>) 
2. TheVirtualTailor: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style [PR20](<https://virtualhumans.mpi-inf.mpg.de/papers/patel20vtailor/vtailor.pdf>)  [Code](<https://virtualhumans.mpi-inf.mpg.de/vtailor/>) 
3. Learning to Dress 3D People in Generative Clothing [PR20](<https://arxiv.org/pdf/1907.13615.pdf>) 
4. Towards Photo-Realistic Virtual Try-On by Adaptively Generating $\leftrightarrow $ Preserving Image Content [PR20](<https://arxiv.org/pdf/2003.05863.pdf>) 

------

### Graph

1. OccuSeg: Occupancy-aware 3D Instance Segmentation [PR20](<https://arxiv.org/pdf/2003.06537.pdf>) [C/S](#Classification/Segmentation) 
2. PointGNN: Graph Neural Network for 3D Object Detection in a Point Cloud [PR20](<https://arxiv.org/pdf/2003.01251.pdf>)  [TF](<https://github.com/WeijingShi/Point-GNN>) [Detection](#Detection/Drive) 
3. HOPENet: A Graph-based Model for Hand-Object Pose Estimation [PR20](https://arxiv.org/pdf/2004.00060.pdf) [Project](http://vision.sice.indiana.edu/projects/hopenet) [Torch](https://github.com/bardiadoosti/HOPE/) [Account](https://mp.weixin.qq.com/s/SedoU-W2wUNIqqmzKArf1A) [Pose Estimation](#Pose Estimation) 
4. THF3D: Towards High-Fidelity 3D Face Reconstruction from In-the-Wild Images Using Graph Convolutional Networks [PR20](<https://arxiv.org/pdf/2003.05653.pdf>) [Reconstruction](#Generation/Reconstruction)
5. GridGCN: Grid-GCN for Fast and Scalable Point Cloud Learning [PR20](<https://arxiv.org/pdf/1912.02984.pdf>) [Account](<https://leijiezhang001.github.io/paper-reading-Grid-GCN-for-Fast-and-Scalable-Point-Cloud-Learning/>) [C/S](#Classification/Segmentation)

------

### Datasets

1. SPARE3D: A Dataset for Spatial Reasoning on Three-View Line Drawings [PR20](https://arxiv.org/pdf/2003.14034.pdf) [Project](https://ai4ce.github.io/SPARE3D/) 
2. Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS [PR20](<https://arxiv.org/pdf/2003.03972.pdf>) 
3. IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning [PR20](<https://arxiv.org/pdf/2003.02920.pdf>) 

------

###  Others

1. StyleRig: Rigging StyleGAN for 3D Control over Portrait Images [PR20](<https://gvv.mpi-inf.mpg.de/projects/StyleRig/data/paper.pdf>) [Project](https://gvv.mpi-inf.mpg.de/projects/StyleRig/) 
2. Neural Pose Transfer By Spatially Adaptive Instance Normalization [PR20](https://arxiv.org/pdf/2003.07254.pdf) [Code](<https://github.com/jiashunwang/Neural-Pose-Transfer>)  [Account](https://mp.weixin.qq.com/s/r5kwqwMzqOvMgLV8cNRNxg) 
3. GeoDA: A Geometric Framework for Black-box Adversarial Attacks [PR20](<https://arxiv.org/pdf/2003.06468.pdf>) 
4. Neural Contours: Learning to Draw Lines from 3D Shapes [PR20](https://arxiv.org/pdf/2003.10333.pdf) 
5. Robust 3D Self-portraits in Seconds [PR20 Oral](https://arxiv.org/pdf/2004.02460.pdf) 
6. Leveraging 2D Data to Learn Textured 3D Mesh Generation [PR20 Oral](https://arxiv.org/pdf/2004.04180.pdf) 
7. X3D: Expanding Architectures for Efficient Video Recognition [PR20 Oral](https://arxiv.org/pdf/2004.04730)  [Code](https://github.com/facebookresearch/SlowFast) 
8. 3D Photography using Context-aware Layered Depth Inpainting [PR20](https://arxiv.org/pdf/2004.04727.pdf) [Code](https://github.com/vt-vl-lab/3d-photo-inpainting) 

------


