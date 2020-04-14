### Classification/Segmentation

---

##### [1] PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation [PR20](https://arxiv.org/pdf/2004.01658.pdf) 

![PointGroupT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PointGroupT.png)

Instance segmentation is an important task for scene understanding. Compared to the fully-developed 2D, 3D instance segmentation for point clouds have much room to improve. In this paper, we present PointGroup, a new end-to-end bottom-up architecture, specifically focused on better grouping the points by exploring the void space between objects. We design a two-branch network to extract point features and predict semantic labels and offsets, for shifting each point towards its respective instance centroid. A clustering component is followed to utilize both the original and offset-shifted point coordinate sets, taking advantage of their complementary strength. Further, we formulate the ScoreNet to evaluate the candidate instances, followed by the Non-Maximum Suppression (NMS) to remove duplicates. We conduct extensive experiments on two challenging datasets, ScanNet v2 and S3DIS, on which our method achieves the highest performance, 63.6% and 64.0%, compared to 54.9% and 54.4% achieved by former best solutions in terms of mAP with IoU threshold 0.5.

![PointGroupfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PointGroupfig2.png)

---

##### [2] Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds [PR20](https://arxiv.org/pdf/2003.12971.pdf) [Torch](https://github.com/raoyongming/PointGLR ) 

![GLBRT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/GLBRT.png)

Local and global patterns of an object are closely related. Although each part of an object is incomplete, the underlying attributes about the object are shared among all parts, which makes reasoning the whole object from a single part possible. We hypothesize that a powerful representation of a 3D object should model the attributes that are shared between parts and the whole object, and distinguishable from other objects. Based on this hypothesis, we propose to learn point cloud representation by bidirectional reasoning between the local structures at different abstraction hierarchies and the global shape without human supervision. Experimental results on various benchmark datasets demonstrate the unsupervisedly learned representation is even better than supervised representation in discriminative power, generalization ability, and robustness. We show that unsupervisedly trained point cloud models can outperform their supervised counterparts on downstream classification tasks. Most notably, by simply increasing the channel width of an SSG PointNet++, our unsupervised model surpasses the state-of-the-art supervised methods on both synthetic and real-world 3D object classification datasets. We expect our observations to offer a new perspective on learning better representation from data structures instead of human annotations for point cloud understanding.

![GLBRfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/GLBRfig2.png)

---

##### [3] 3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation [PR20](https://arxiv.org/pdf/2003.13867.pdf) [Project](https://www.vision.rwth-aachen.de/publication/00199/) [Video](https://youtu.be/ifL8yTbRFDk) 

![3DMPAT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/3DMPAT.png)

We present 3D-MPA, a method for instance segmentation on 3D point clouds. Given an input point cloud, we propose an object-centric approach where each point votes for its object center. We sample object proposals from the predicted object centers. Then, we learn proposal features from grouped point features that voted for the same object center. A graph convolutional network introduces inter-proposal relations, providing higher-level feature learning in addition to the lower-level point features. Each proposal comprises a semantic label, a set of associated points over which we define a foreground-background mask, an objectness score and aggregation features. Previous works usually perform non-maximum-suppression (NMS) over proposals to obtain the final object detections or semantic instances. However, NMS can discard potentially correct predictions. Instead, our approach keeps all proposals and groups them together based on the learned aggregation features. We show that grouping proposals improves over NMS and outperforms previous state-of-the-art methods on the tasks of 3D object detection and semantic instance segmentation on the ScanNetV2 benchmark and the S3DIS dataset.

![3DMPAfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/3DMPAfig2.png)

---

##### [4] PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation [PR20](https://arxiv.org/pdf/2003.14032.pdf) [Torch](https://github.com/edwardzhou130/PolarSeg) 

![PolarNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PolarNetT.png)

The requirement of fine-grained perception by autonomous driving systems has resulted in recently increased research in the online semantic segmentation of single-scan LiDAR. Emerging datasets and technological advancements have enabled researchers to benchmark this problem and improve the applicable semantic segmentation algorithms. Still, online semantic segmentation of LiDAR scans in autonomous driving applications remains challenging due to three reasons: (1) the need for near-real-time latency with limited hardware, (2) points are distributed unevenly across space, and (3) an increasing number of more fine-grained semantic classes. The combination of the aforementioned challenges motivates us to propose a new LiDAR-specific, KNN-free segmentation algorithm - PolarNet. Instead of using common spherical or bird's-eye-view projection, our polar bird's-eye-view representation balances the points per grid and thus indirectly redistributes the network's attention over the long-tailed points distribution over the radial axis in polar coordination. We find that our encoding scheme greatly increases the mIoU in three drastically different real urban LiDAR single-scan segmentation datasets while retaining ultra low latency and near real-time throughput.

![PolarNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PolarNetfig2.png)

---

##### [5] DualConvMeshNet: Joint Geodesic and Euclidean Convolutions on 3D Meshes [PR20](https://arxiv.org/pdf/2004.01002.pdf) [Torch](https://github.com/VisualComputingInstitute/dcm-net) 

![DCMNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/DCMNT.png)

We propose DualConvMesh-Nets (DCM-Net) a family of deep hierarchical convolutional networks over 3D geometric data that combines two types of convolutions. The first type, geodesic convolutions, defines the kernel weights over mesh surfaces or graphs. That is, the convolutional kernel weights are mapped to the local surface of a given mesh. The second type, Euclidean convolutions, is independent of any underlying mesh structure. The convolutional kernel is applied on a neighborhood obtained from a local affinity representation based on the Euclidean distance between 3D points. Intuitively, geodesic convolutions can easily separate objects that are spatially close but have disconnected surfaces, while Euclidean convolutions can represent interactions between nearby objects better, as they are oblivious to object surfaces. To realize a multi-resolution architecture, we borrow well-established mesh simplification methods from the geometry processing domain and adapt them to define mesh-preserving pooling and unpooling operations. We experimentally show that combining both types of convolutions in our architecture leads to significant performance gains for 3D semantic segmentation, and we report competitive results on three scene segmentation benchmarks. Our models and code are publicly available.

![DCMNfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/DCMNfig3.png)

---

##### [6] Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image [PR20](https://arxiv.org/pdf/2004.01176.pdf) [Torch](https://github.com/paschalidoud/hierarchical_primitives) 

![LUHPDT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/LUHPDT.png)

Humans perceive the 3D world as a set of distinct objects that are characterized by various low-level (geometry, reflectance) and high-level (connectivity, adjacency, symmetry) properties. Recent methods based on convolutional neural networks (CNNs) demonstrated impressive progress in 3D reconstruction, even when using a single 2D image as input. However, the majority of these methods focuses on recovering the local 3D geometry of an object without considering its part-based decomposition or relations between parts. We address this challenging problem by proposing a novel formulation that allows to jointly recover the geometry of a 3D object as a set of primitives as well as their latent hierarchical structure without part-level supervision. Our model recovers the higher level structural decomposition of various objects in the form of a binary tree of primitives, where simple parts are represented with fewer primitives and more complex parts are modeled with more components. Our experiments on the ShapeNet and D-FAUST datasets demonstrate that considering the organization of parts indeed facilitates reasoning about 3D geometry.

![LUHPDfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/LUHPDfig3.png)

---

##### [7] Multi-Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds [PR20](https://arxiv.org/pdf/2003.13035.pdf) 

![MPRMT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/MPRMT.png)

Point clouds provide intrinsic geometric information and surface context for scene understanding. Existing methods for point cloud segmentation require a large amount of fully labeled data. Using advanced depth sensors, collection of large scale 3D dataset is no longer a cumbersome process. However, manually producing point-level label on the large scale dataset is time and labor-intensive. In this paper, we propose a weakly supervised approach to predict point-level results using weak labels on 3D point clouds. We introduce our multi-path region mining module to generate pseudo point-level label from a classification network trained with weak labels. It mines the localization cues for each class from various aspects of the network feature using different attention modules. Then, we use the point-level pseudo labels to train a point cloud segmentation network in a fully supervised manner. To the best of our knowledge, this is the first method that uses cloud-level weak labels on raw 3D space to train a point cloud semantic segmentation network. In our setting, the 3D weak labels only indicate the classes that appeared in our input sample. We discuss both scene- and subcloud-level weakly labels on raw 3D point cloud data and perform in-depth experiments on them. On ScanNet dataset, our result trained with subcloud-level labels is compatible with some fully supervised methods.

![MPRMfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/MPRMfig3.png)

---

##### [8] Learning to Segment 3D Point Clouds in 2D Image Space [PR20](<https://arxiv.org/pdf/2003.05593.pdf>) [TF](<https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space>)

![LST](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/LST.png)

In contrast to the literature where local patterns in 3D point clouds are captured by customized convolutional operators, in this paper we study the problem of how to effectively and efficiently project such point clouds into a 2D image space so that traditional 2D convolutional neural networks (CNNs) such as U-Net can be applied for segmentation. To this end, we are motivated by graph drawing and reformulate it as an integer programming problem to learn the topology-preserving graph-to-grid mapping for each individual point cloud. To accelerate the computation in practice, we further propose a novel hierarchical approximate algorithm. With the help of the Delaunay triangulation for graph construction from point clouds and a multi-scale U-Net for segmentation, we manage to demonstrate the state-of-the-art performance on ShapeNet and PartNet, respectively, with significant improvement over the literature.

![LSfig4](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/LSfig4.png)

---

##### [9] xMUDA： Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [PR20](https://arxiv.org/abs/1911.12676) [Code](<https://github.com/valeoai/xmuda>) [Video](<http://tiny.cc/xmuda>)  

![xMuDAT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/xMuDAT.png)

Unsupervised Domain Adaptation (UDA) is crucial to tackle the lack of annotations in a new domain. There are many multi-modal datasets, but most UDA approaches are uni-modal. In this work, we explore how to learn from multi-modality and propose cross-modal UDA (xMUDA) where we assume the presence of 2D images and 3D point clouds for 3D semantic segmentation. This is challenging as the two input spaces are heterogeneous and can be impacted differently by domain shift. In xMUDA, modalities learn from each other through mutual mimicking, disentangled from the segmentation objective, to prevent the stronger modality from adopting false predictions from the weaker one. We evaluate on new UDA scenarios including day-to-night, country-to-country and dataset-to-dataset, leveraging recent autonomous driving datasets. xMUDA brings large improvements over uni-modal UDA on all tested scenarios, and is complementary to state-of-the-art UDA techniques.

![xMUDAfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/xMUDAfig2.png)

----

##### [10] FPConv: Learning Local Flattening for Point Convolution [PR20](<https://arxiv.org/pdf/2002.10701.pdf>) [Torch](<https://github.com/lyqun/FPConv>) 

![FPConvT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/FPConvT.png)

We introduce FPConv, a novel surface-style convolution operator designed for 3D point cloud analysis. Unlike previous methods, FPConv doesn't require transforming to intermediate representation like 3D grid or graph and directly works on surface geometry of point cloud. To be more specific, for each point, FPConv performs a local flattening by automatically learning a weight map to softly project surrounding points onto a 2D grid. Regular 2D convolution can thus be applied for efficient feature learning. FPConv can be easily integrated into various network architectures for tasks like 3D object classification and 3D scene segmentation, and achieve comparable performance with existing volumetric-type convolutions. More importantly, our experiments also show that FPConv can be a complementary of volumetric convolutions and jointly training them can further boost overall performance into state-of-the-art results.

![FPConvfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/FPConvfig3.png)

---

##### [11] OccuSeg: Occupancy-aware 3D Instance Segmentation [PR20](<https://arxiv.org/pdf/2003.06537.pdf>) [Graph]

![OccuSegT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/OccuSegT.png)

3D instance segmentation, with a variety of applications in robotics and augmented reality, is in large demands these days. Unlike 2D images that are projective observations of the environment, 3D models provide metric reconstruction of the scenes without occlusion or scale ambiguity. In this paper, we define "3D occupancy size", as the number of voxels occupied by each instance. It owns advantages of robustness in prediction, on which basis, OccuSeg, an occupancy-aware 3D instance segmentation scheme is proposed. Our multi-task learning produces both occupancy signal and embedding representations, where the training of spatial and feature embeddings varies with their difference in scale-aware. Our clustering scheme benefits from the reliable comparison between the predicted occupancy size and the clustered occupancy size, which encourages hard samples being correctly clustered and avoids over segmentation. The proposed approach achieves state-of-the-art performance on 3 real-world datasets, i.e. ScanNetV2, S3DIS and SceneNN, while maintaining high efficiency.

![OccuSegfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/OccuSegfig2.png)

---

##### [12] GridGCN: Grid-GCN for Fast and Scalable Point Cloud Learning [PR20](<https://arxiv.org/pdf/1912.02984.pdf>) [Account](<https://leijiezhang001.github.io/paper-reading-Grid-GCN-for-Fast-and-Scalable-Point-Cloud-Learning/>) [Graph]

![GridGCNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/GridGCNT.png)

Due to the sparsity and irregularity of the point cloud data, methods that directly consume points have become popular. Among all point-based models, graph convolutional networks (GCN) lead to notable performance by fully preserving the data granularity and exploiting point interrelation. However, point-based networks spend a significant amount of time on data structuring (e.g., Farthest Point Sampling (FPS) and neighbor points querying), which limit the speed and scalability. In this paper, we present a method, named Grid-GCN, for fast and scalable point cloud learning. Grid-GCN uses a novel data structuring strategy, Coverage-Aware Grid Query (CAGQ). By leveraging the efficiency of grid space, CAGQ improves spatial coverage while reducing the theoretical time complexity. Compared with popular sampling methods such as Farthest Point Sampling (FPS) and Ball Query, CAGQ achieves up to 50X speed-up. With a Grid Context Aggregation (GCA) module, Grid-GCN achieves state-of-the-art performance on major point cloud classification and segmentation benchmarks with significantly faster runtime than previous studies. Remarkably, Grid-GCN achieves the inference speed of 50fps on ScanNet using 81920 points per scene as input.

![GridGCNfig4](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/GridGCNfig4.png)

---

##### [13] PointAugment: An Auto-Augmentation Framework for Point Cloud Classification [PR20](<https://arxiv.org/pdf/2002.10876.pdf>) [Code](<https://github.com/liruihui/PointAugment/>) 

![PointAugmentT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PointAugmentT.png)

We present PointAugment, a new auto-augmentation framework that automatically optimizes and augments point cloud samples to enrich the data diversity when we train a classification network. Different from existing auto-augmentation methods for 2D images, PointAugment is sample-aware and takes an adversarial learning strategy to jointly optimize an augmentor network and a classifier network, such that the augmentor can learn to produce augmented samples that best fit the classifier. Moreover, we formulate a learnable point augmentation function with a shape-wise transformation and a point-wise displacement, and carefully design loss functions to adopt the augmented samples based on the learning progress of the classifier. Extensive experiments also confirm PointAugment's effectiveness and robustness to improve the performance of various networks on shape classification and retrieval.

![PointAugmentfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PointAugmentfig2.png)

---

##### [14] PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling [PR20](<https://arxiv.org/pdf/2003.00492.pdf>) [Code](https://github.com/yanx27/PointASNL)

![PointASNLT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PointASNLT.png)

Raw point clouds data inevitably contains outliers or noise through acquisition from 3D sensors or reconstruction algorithms. In this paper, we present a novel end-to-end network for robust point clouds processing, named PointASNL, which can deal with point clouds with noise effectively. The key component in our approach is the adaptive sampling (AS) module. It first re-weights the neighbors around the initial sampled points from farthest point sampling (FPS), and then adaptively adjusts the sampled points beyond the entire point cloud. Our AS module can not only benefit the feature learning of point clouds, but also ease the biased effect of outliers. To further capture the neighbor and long-range dependencies of the sampled point, we proposed a local-nonlocal (L-NL) module inspired by the nonlocal operation. Such L-NL module enables the learning process insensitive to noise. Extensive experiments verify the robustness and superiority of our approach in point clouds processing tasks regardless of synthesis data, indoor data, and outdoor data with or without noise. Specifically, PointASNL achieves state-of-the-art robust performance for classification and segmentation tasks on all datasets, and significantly outperforms previous methods on real-world outdoor SemanticKITTI dataset with considerate noise.

![PointASNLfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/PointASNLfig2.png)

---

##### [15] RandLANet: Efficient Semantic Segmentation of Large-Scale Point Clouds [PR20](<https://arxiv.org/pdf/1911.11236.pdf>) [TF](<https://github.com/QingyongHu/RandLA-Net>) [Account](<https://zhuanlan.zhihu.com/p/105433460>)

![RandLANetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/RandLANetT.png)

We study the problem of efficient semantic segmentation for large-scale 3D point clouds. By relying on expensive sampling techniques or computationally heavy pre/post-processing steps, most existing approaches are only able to be trained and operate over small-scale point clouds. In this paper, we introduce RandLA-Net, an efficient and lightweight neural architecture to directly infer per-point semantics for large-scale point clouds. The key to our approach is to use random point sampling instead of more complex point selection approaches. Although remarkably computation and memory efficient, random sampling can discard key features by chance. To overcome this, we introduce a novel local feature aggregation module to progressively increase the receptive field for each 3D point, thereby effectively preserving geometric details. Extensive experiments show that our RandLA-Net can process 1 million points in a single pass with up to 200X faster than existing approaches. Moreover, our RandLA-Net clearly surpasses state-of-the-art approaches for semantic segmentation on two large-scale benchmarks Semantic3D and SemanticKITTI.

![RandLANetfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/RandLANetfig3.png)

---

##### [16] Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels [PR20](<https://arxiv.org/pdf/2004.04091.pdf>) [Code](<https://github.com/alex-xun-xu/WeakSupPointCloudSeg>) 

![WSST](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/WSST.png)

Point cloud analysis has received much attention recently; and segmentation is one of the most important tasks. The success of existing approaches is attributed to deep network design and large amount of labelled training data, where the latter is assumed to be always available. However, obtaining 3d point cloud segmentation labels is often very costly in practice. In this work, we propose a weakly supervised point cloud segmentation approach which requires only a tiny fraction of points to be labelled in the training stage. This is made possible by learning gradient approximation and exploitation of additional spatial and color smoothness constraints. Experiments are done on three public datasets with different degrees of weak supervision. In particular, our proposed method can produce results that are close to and sometimes even better than its fully supervised counterpart with 10× fewer labels.

![WSSfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Classification%20Segmentation/WSSfig2.png)

------

