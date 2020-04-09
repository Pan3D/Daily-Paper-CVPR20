### Matching

##### [1] RPMNet: Robust Point Matching using Learned Features  [PR20](https://arxiv.org/pdf/2003.13479.pdf)  [Code](https://github.com/yewzijian/RPMNet)  

![RPMNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/RPMNetT.png)

Iterative Closest Point (ICP) solves the rigid point cloud registration problem iteratively in two steps: (1) make hard assignments of spatially closest point correspondences, and then (2) find the least-squares rigid transformation. The hard assignments of closest point correspondences based on spatial distances are sensitive to the initial rigid transformation and noisy/outlier points, which often cause ICP to converge to wrong local minima. In this paper, we propose the RPM-Net -- a less sensitive to initialization and more robust deep learning-based approach for rigid point cloud registration. To this end, our network uses the differentiable Sinkhorn layer and annealing to get soft assignments of point correspondences from hybrid features learned from both spatial coordinates and local geometry. To further improve registration performance, we introduce a secondary network to predict optimal annealing parameters. Unlike some existing methods, our RPM-Net handles missing correspondences and point clouds with partial visibility. Experimental results show that our RPM-Net achieves state-of-the-art performance compared to existing non-deep learning and recent deep learning methods. 

![RPMNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/RPMNetfig2.png)

---

##### [2] Learning Multiview 3D Point Cloud Registration [PR20](<https://arxiv.org/pdf/2001.05119.pdf>) [Code](<https://github.com/zgojcic/3D_multiview_reg>) 

![LMVT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/LMVT.png)

We present a novel, end-to-end learnable, multiview 3D point cloud registration algorithm. Registration of multiple scans typically follows a two-stage pipeline: the initial pairwise alignment and the globally consistent refinement. The former is often ambiguous due to the low overlap of neighboring point clouds, symmetries and repetitive scene parts. Therefore, the latter global refinement aims at establishing the cyclic consistency across multiple scans and helps in resolving the ambiguous cases. In this paper we propose, to the best of our knowledge, the first end-to-end algorithm for joint learning of both parts of this two-stage problem. Experimental evaluation on well accepted benchmark datasets shows that our approach outperforms the state-of-the-art by a significant margin, while being end-to-end trainable and computationally less costly. Moreover, we present detailed analysis and an ablation study that validate the novel components of our approach.

![LMVfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/LMVfig2.png)

---

##### [3] D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features [PR20](<https://arxiv.org/pdf/2003.03164.pdf>) [TF](<https://github.com/XuyangBai/D3Feat>) 

![D3FeatT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/D3FeatT.png)

A successful point cloud registration often lies on robust establishment of sparse matches through discriminative 3D local features. Despite the fast evolution of learning-based 3D feature descriptors, little attention has been drawn to the learning of 3D feature detectors, even less for a joint learning of the two tasks. In this paper, we leverage a 3D fully convolutional network for 3D point clouds, and propose a novel and practical learning mechanism that densely predicts both a detection score and a description feature for each 3D point. In particular, we propose a keypoint selection strategy that overcomes the inherent density variations of 3D point clouds, and further propose a self-supervised detector loss guided by the on-the-fly feature matching results during training. Finally, our method achieves state-of-the-art results in both indoor and outdoor scenarios, evaluated on 3DMatch and KITTI datasets, and shows its strong generalization ability on the ETH dataset. Towards practical use, we show that by adopting a reliable feature detector, sampling a smaller number of features is sufficient to achieve accurate and fast point cloud alignment.

![D3Featfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/D3Featfig1.png)

---

##### [4] End-to-End Learning Local Multi-view Descriptors for 3D Point Clouds [PR20](<https://arxiv.org/pdf/2003.05855.pdf>) 

![E2ELLMT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/E2ELLMT.png)

In this work, we propose an end-to-end framework to learn local multi-view descriptors for 3D point clouds. To adopt a similar multi-view representation, existing studies use hand-crafted viewpoints for rendering in a preprocessing stage, which is detached from the subsequent descriptor learning stage. In our framework, we integrate the multi-view rendering into neural networks by using a differentiable renderer, which allows the viewpoints to be optimizable parameters for capturing more informative local context of interest points. To obtain discriminative descriptors, we also design a soft-view pooling module to attentively fuse convolutional features across views. Extensive experiments on existing 3D registration benchmarks show that our method outperforms existing local descriptors both quantitatively and qualitatively.

![E2ELLMfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/E2ELLMfig1.png)

---

### Graph

##### [5] OccuSeg: Occupancy-aware 3D Instance Segmentation [PR20](<https://arxiv.org/pdf/2003.06537.pdf>) [C/S](#Classification/Segmentation) 

![OccuSegT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/OccuSegT.png)

3D instance segmentation, with a variety of applications in robotics and augmented reality, is in large demands these days. Unlike 2D images that are projective observations of the environment, 3D models provide metric reconstruction of the scenes without occlusion or scale ambiguity. In this paper, we define "3D occupancy size", as the number of voxels occupied by each instance. It owns advantages of robustness in prediction, on which basis, OccuSeg, an occupancy-aware 3D instance segmentation scheme is proposed. Our multi-task learning produces both occupancy signal and embedding representations, where the training of spatial and feature embeddings varies with their difference in scale-aware. Our clustering scheme benefits from the reliable comparison between the predicted occupancy size and the clustered occupancy size, which encourages hard samples being correctly clustered and avoids over segmentation. The proposed approach achieves state-of-the-art performance on 3 real-world datasets, i.e. ScanNetV2, S3DIS and SceneNN, while maintaining high efficiency.

![OccuSegfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/OccuSegfig2.png)

---

##### [6] PointGNN: Graph Neural Network for 3D Object Detection in a Point Cloud [PR20](<https://arxiv.org/pdf/2003.01251.pdf>)  [TF](<https://github.com/WeijingShi/Point-GNN>) [Detection] 

![PointGNNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/PointGNNT.png)

In this paper, we propose a graph neural network to detect objects from a LiDAR point cloud. Towards this end, we encode the point cloud efficiently in a fixed radius near-neighbors graph. We design a graph neural network, named Point-GNN, to predict the category and shape of the object that each vertex in the graph belongs to. In Point-GNN, we propose an auto-registration mechanism to reduce translation variance, and also design a box merging and scoring operation to combine detections from multiple vertices accurately. Our experiments on the KITTI benchmark show the proposed approach achieves leading accuracy using the point cloud alone and can even surpass fusion-based algorithms. Our results demonstrate the potential of using the graph neural network as a new approach for 3D object detection.

![PointGNNfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/PointGNNfig2.png)

---

##### [7] HOPENet: A Graph-based Model for Hand-Object Pose Estimation [PR20](https://arxiv.org/pdf/2004.00060.pdf) [Project](http://vision.sice.indiana.edu/projects/hopenet) [Torch](https://github.com/bardiadoosti/HOPE/) [Account](https://mp.weixin.qq.com/s/SedoU-W2wUNIqqmzKArf1A) [Pose Estimation] 

![HOPENetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/HOPENetT.png)

Hand-object pose estimation (HOPE) aims to jointly detect the poses of both a hand and of a held object. In this paper, we propose a lightweight model called HOPE-Net which jointly estimates hand and object pose in 2D and 3D in real-time. Our network uses a cascade of two adaptive graph convolutional neural networks, one to estimate 2D coordinates of the hand joints and object corners, followed by another to convert 2D coordinates to 3D. Our experiments show that through end-to-end training of the full network, we achieve better accuracy for both the 2D and 3D coordinate estimation problems. The proposed 2D to 3D graph convolution-based model could be applied to other 3D landmark detection problems, where it is possible to first predict the 2D keypoints and then transform them to 3D.

![HOPENetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/HOPENetfig2.png)

---

##### [8] THF3D: Towards High-Fidelity 3D Face Reconstruction from In-the-Wild Images Using Graph Convolutional Networks [PR20](<https://arxiv.org/pdf/2003.05653.pdf>) [Reconstruction]

![THFT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/THFT.png)

3D Morphable Model (3DMM) based methods have achieved great success in recovering 3D face shapes from single-view images. However, the facial textures recovered by such methods lack the fidelity as exhibited in the input images. Recent work demonstrates high-quality facial texture recovering with generative networks trained from a large-scale database of high-resolution UV maps of face textures, which is hard to prepare and not publicly available. In this paper, we introduce a method to reconstruct 3D facial shapes with high-fidelity textures from single-view images in-the-wild, without the need to capture a large-scale face texture database. The main idea is to refine the initial texture generated by a 3DMM based method with facial details from the input image. To this end, we propose to use graph convolutional networks to reconstruct the detailed colors for the mesh vertices instead of reconstructing the UV map. Experiments show that our method can generate high-quality results and outperforms state-of-the-art methods in both qualitative and quantitative comparisons.

![THFfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/THFfig2.png)

---

##### [9] GridGCN: Grid-GCN for Fast and Scalable Point Cloud Learning [PR20](<https://arxiv.org/pdf/1912.02984.pdf>) [Code](<https://github.com/xharlie/Grid-GCN>) [Supp](<https://xharlie.github.io/papers/GGCN_supCamReady.pdf>) [Account](<https://leijiezhang001.github.io/paper-reading-Grid-GCN-for-Fast-and-Scalable-Point-Cloud-Learning/>) [C/S]

![GridGCNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/GridGCNT.png)

Due to the sparsity and irregularity of the point cloud data, methods that directly consume points have become popular. Among all point-based models, graph convolutional networks (GCN) lead to notable performance by fully preserving the data granularity and exploiting point interrelation. However, point-based networks spend a significant amount of time on data structuring (e.g., Farthest Point Sampling (FPS) and neighbor points querying), which limit the speed and scalability. In this paper, we present a method, named Grid-GCN, for fast and scalable point cloud learning. Grid-GCN uses a novel data structuring strategy, Coverage-Aware Grid Query (CAGQ). By leveraging the efficiency of grid space, CAGQ improves spatial coverage while reducing the theoretical time complexity. Compared with popular sampling methods such as Farthest Point Sampling (FPS) and Ball Query, CAGQ achieves up to 50X speed-up. With a Grid Context Aggregation (GCA) module, Grid-GCN achieves state-of-the-art performance on major point cloud classification and segmentation benchmarks with significantly faster runtime than previous studies. Remarkably, Grid-GCN achieves the inference speed of 50fps on ScanNet using 81920 points per scene as input.

![GridGCNfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/GridGCNfig2.png)

----

### Others

##### [10] StyleRig: Rigging StyleGAN for 3D Control over Portrait Images [PR20](<https://gvv.mpi-inf.mpg.de/projects/StyleRig/data/paper.pdf>) [Project](https://gvv.mpi-inf.mpg.de/projects/StyleRig/) 

![StyleRigT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/StyleRigT.png)

StyleGAN generates photorealistic portrait images of faces with eyes, teeth, hair and context (neck, shoulders, background), but lacks a rig-like control over semantic face
parameters that are interpretable in 3D, such as face pose, expressions, and scene illumination. Three-dimensional morphable face models (3DMMs) on the other hand offer control over the semantic parameters, but lack photorealism when rendered and only model the face interior, not other parts of a portrait image (hair, mouth interior, background). We present the first method to provide a face rig-like control over a pretrained and fixed StyleGAN via a 3DMM. A new rigging network, RigNet is trained between the 3DMM’s semantic parameters and StyleGAN’s input. The network is trained in a self-supervised manner, without the need for manual annotations. At test time, our method generates portrait images with the photorealism of StyleGAN and provides explicit control over the 3D semantic parameters of the face.

![StyleRigfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/StyleRigfig2.png)

---

##### [11] Neural Pose Transfer By Spatially Adaptive Instance Normalization [PR20](https://arxiv.org/pdf/2003.07254.pdf) [Code](<https://github.com/jiashunwang/Neural-Pose-Transfer>) [Account](https://mp.weixin.qq.com/s/r5kwqwMzqOvMgLV8cNRNxg) 

![NPTT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/NPTT.png)

Pose transfer has been studied for decades, in which the pose of a source mesh is applied to a target mesh. Particularly in this paper, we are interested in transferring the pose of source human mesh to deform the target human mesh, while the source and target meshes may have different identity information. Traditional studies assume that the paired source and target meshes are existed with the point-wise correspondences of user annotated landmarks/mesh points, which requires heavy labelling efforts. On the other hand, the generalization ability of deep models is limited, when the source and target meshes have different identities. To break this limitation, we proposes the first neural pose transfer model that solves the pose transfer via the latest technique for image style transfer, leveraging the newly proposed component -- spatially adaptive instance normalization. Our model does not require any correspondences between the source and target meshes. Extensive experiments show that the proposed model can effectively transfer deformation from source to target meshes, and has good generalization ability to deal with unseen identities or poses of meshes. 

![NPTfig4](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/NPTfig4.png)

---

##### [12] GeoDA: A Geometric Framework for Black-box Adversarial Attacks [PR20](<https://arxiv.org/pdf/2003.06468.pdf>) 

![GeoDAT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/GeoDAT.png)

Adversarial examples are known as carefully perturbed images fooling image classifiers. We propose a geometric framework to generate adversarial examples in one of the most challenging black-box settings where the adversary can only generate a small number of queries, each of them returning the top-1 label of the classifier. Our framework is based on the observation that the decision boundary of deep networks usually has a small mean curvature in the vicinity of data samples. We propose an effective iterative algorithm to generate query-efficient black-box perturbations with small ℓp norms for p≥1, which is confirmed via experimental evaluations on state-of-the-art natural image classifiers. Moreover, for p=2, we theoretically show that our algorithm actually converges to the minimal ℓ2-perturbation when the curvature of the decision boundary is bounded. We also obtain the optimal distribution of the queries over the iterations of the algorithm. Finally, experimental results confirm that our principled black-box attack algorithm performs better than state-of-the-art algorithms as it generates smaller perturbations with a reduced number of queries.

![GeoDAfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/GeoDAfig1.png)

---

##### [13] Neural Contours: Learning to Draw Lines from 3D Shapes [PR20](https://arxiv.org/pdf/2003.10333.pdf) 

![NCT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/NCT.png)

This paper introduces a method for learning to generate line drawings from 3D models. Our architecture incorporates a differentiable module operating on geometric features of the 3D model, and an image-based module operating on view-based shape representations. At test time, geometric and view-based reasoning are combined with the help of a neural module to create a line drawing. The model is trained on a large number of crowdsourced comparisons of line drawings. Experiments demonstrate that our method achieves significant improvements in line drawing over the state-of-the-art when evaluated on standard benchmarks, resulting in drawings that are comparable to those produced by experienced human artists.

![NCfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/NCfig3.png)

---

##### [14] Robust 3D Self-portraits in Seconds [PR20 Oral](https://arxiv.org/pdf/2004.02460.pdf) 

![R3DSPT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/R3DSPT.png)

In this paper, we propose an efficient method for robust 3D self-portraits using a single RGBD camera. Benefiting from the proposed PIFusion and lightweight bundle adjustment algorithm, our method can generate detailed 3D self-portraits in seconds and shows the ability to handle subjects wearing extremely loose clothes. To achieve highly efficient and robust reconstruction, we propose PIFusion, which combines learning-based 3D recovery with volumetric non-rigid fusion to generate accurate sparse partial scans of the subject. Moreover, a non-rigid volumetric deformation method is proposed to continuously refine the learned shape prior. Finally, a lightweight bundle adjustment algorithm is proposed to guarantee that all the partial scans can not only "loop" with each other but also remain consistent with the selected live key observations. The results and experiments show that the proposed method achieves more robust and efficient 3D self-portraits compared with state-of-the-art methods.

![R3DSPfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Face%2C%20Graph%2C%20Others/R3DSPfig2.png)

------

