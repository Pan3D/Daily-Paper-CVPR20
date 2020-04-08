### Pose Estimation

---

#### [1] Towards Better Generalization: Joint Depth-Pose Learning without PoseNet [PR20](https://arxiv.org/pdf/2004.01314.pdf) [Torch](https://github.com/B1ueber2y/TrianFlow) [Depth](#Depth)

![TBGT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/TBGT.png)

In this work, we tackle the essential problem of scale inconsistency for self-supervised joint depth-pose learning. Most existing methods assume that a consistent scale of depth and pose can be learned across all input samples, which makes the learning problem harder, resulting in degraded performance and limited generalization in indoor environments and long-sequence visual odometry application. To address this issue, we propose a novel system that explicitly disentangles scale from the network estimation. Instead of relying on PoseNet architecture, our method recovers relative pose by directly solving fundamental matrix from dense optical flow correspondence and makes use of a two-view triangulation module to recover an up-to-scale 3D structure. Then, we align the scale of the depth prediction with the triangulated point cloud and use the transformed depth map for depth error computation and dense reprojection check. Our whole system can be jointly trained end-to-end. Extensive experiments show that our system not only reaches state-of-the-art performance on KITTI depth and flow estimation, but also significantly improves the generalization ability of existing self-supervised depth-pose learning methods under a variety of challenging scenarios, and achieves state-of-the-art results among self-supervised learning-based methods on KITTI Odometry and NYUv2 dataset. Furthermore, we present some interesting findings on the limitation of PoseNet-based relative pose estimation methods in terms of generalization ability. 

![TBGfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/TBGfig2.png)

---

#### [2] HandVoxNet: Deep Voxel-Based Network for 3D Hand Shape and Pose Estimation from a Single Depth Map [PR20](https://arxiv.org/pdf/2004.01588.pdf) 

![HandVoxNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/HandVoxNetT.png)

3D hand shape and pose estimation from a single depth map is a new and challenging computer vision problem with many applications. The state-of-the-art methods directly regress 3D hand meshes from 2D depth images via 2D convolutional neural networks, which leads to artefacts in the estimations due to perspective distortions in the images. In contrast, we propose a novel architecture with 3D convolutions trained in a weakly-supervised manner. The input to our method is a 3D voxelized depth map, and we rely on two hand shape representations. The first one is the 3D voxelized grid of the shape which is accurate but does not preserve the mesh topology and the number of mesh vertices. The second representation is the 3D hand surface which is less accurate but does not suffer from the limitations of the first representation. We combine the advantages of these two representations by registering the hand surface to the voxelized hand shape. In the extensive experiments, the proposed approach improves over the state of the art by 47.8% on the SynHand5M dataset. Moreover, our augmentation policy for voxelized depth maps further enhances the accuracy of 3D hand pose estimation on real data. Our method produces visually more reasonable and realistic hand shapes on NYU and BigHand2.2M datasets compared to the existing approaches.

![HandVoxNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/HandVoxNetfig2.png)

---

#### [3] ONP: Optical Non-Line-of-Sight Physics-based 3D Human Pose Estimation [PR20](https://arxiv.org/pdf/2003.14414.pdf) [Project](https://marikoisogawa.github.io/project/nlos_pose) 

![ONPT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/ONPT.png)

We describe a method for 3D human pose estimation from transient images (i.e., a 3D spatio-temporal histogram of photons) acquired by an optical non-line-of-sight (NLOS) imaging system. Our method can perceive 3D human pose by `looking around corners' through the use of light indirectly reflected by the environment. We bring together a diverse set of technologies from NLOS imaging, human pose estimation and deep reinforcement learning to construct an end-to-end data processing pipeline that converts a raw stream of photon measurements into a full 3D human pose sequence estimate. Our contributions are the design of data representation process which includes (1) a learnable inverse point spread function (PSF) to convert raw transient images into a deep feature vector; (2) a neural humanoid control policy conditioned on the transient image feature and learned from interactions with a physics simulator; and (3) a data synthesis and augmentation strategy based on depth data that can be transferred to a real-world NLOS imaging system. Our preliminary experiments suggest that our method is able to generalize to real-world NLOS measurement to estimate physically-valid 3D human poses.

![ONPfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/ONPfig3.png)

---

#### [4] HOPENet: A Graph-based Model for Hand-Object Pose Estimation [PR20](https://arxiv.org/pdf/2004.00060.pdf) [Project](http://vision.sice.indiana.edu/projects/hopenet) [Torch](https://github.com/bardiadoosti/HOPE/) [Account](https://mp.weixin.qq.com/s/SedoU-W2wUNIqqmzKArf1A) [Graph](#Graph)

![HOPENetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/HOPENetT.png)

Hand-object pose estimation (HOPE) aims to jointly detect the poses of both a hand and of a held object. In this paper, we propose a lightweight model called HOPE-Net which jointly estimates hand and object pose in 2D and 3D in real-time. Our network uses a cascade of two adaptive graph convolutional neural networks, one to estimate 2D coordinates of the hand joints and object corners, followed by another to convert 2D coordinates to 3D. Our experiments show that through end-to-end training of the full network, we achieve better accuracy for both the 2D and 3D coordinate estimation problems. The proposed 2D to 3D graph convolution-based model could be applied to other 3D landmark detection problems, where it is possible to first predict the 2D keypoints and then transform them to 3D.

![HOPENetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/HOPENetfig2.png)

---

#### [5] Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation [PR20](https://arxiv.org/pdf/2004.00329.pdf ) [Torch](https://github.com/fabbrimatteo/LoCO) 

![CVHT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/CVHT.png)

In this paper we present a novel approach for bottom-up multi-person 3D human pose estimation from monocular RGB images. We propose to use high resolution volumetric heatmaps to model joint locations, devising a simple and effective compression method to drastically reduce the size of this representation. At the core of the proposed method lies our Volumetric Heatmap Autoencoder, a fully-convolutional network tasked with the compression of ground-truth heatmaps into a dense intermediate representation. A second model, the Code Predictor, is then trained to predict these codes, which can be decompressed at test time to re-obtain the original representation. Our experimental evaluation shows that our method performs favorably when compared to state of the art on both multi-person and single-person 3D human pose estimation datasets and, thanks to our novel compression strategy, can process full-HD images at the constant runtime of 8 fps regardless of the number of subjects in the scene.

![CVTfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/CVTfig2.png)

---

#### [6] Estimating 6D Pose of Objects with Symmetries [PR20](https://arxiv.org/pdf/2004.00605.pdf) [Project](http://cmp.felk.cvut.cz/epos/ ) 

![EPOST](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/EPOST.png)

We present a new method for estimating the 6D pose of rigid objects with available 3D models from a single RGB input image. The method is applicable to a broad range of objects, including challenging ones with global or partial symmetries. An object is represented by compact surface fragments which allow handling symmetries in a systematic manner. Correspondences between densely sampled pixels and the fragments are predicted using an encoder-decoder network. At each pixel, the network predicts: (i) the probability of each object's presence, (ii) the probability of the fragments given the object's presence, and (iii) the precise 3D location on each fragment. A data-dependent number of corresponding 3D locations is selected per pixel, and poses of possibly multiple object instances are estimated using a robust and efficient variant of the PnP-RANSAC algorithm. In the BOP Challenge 2019, the method outperforms all RGB and most RGB-D and D methods on the T-LESS and LM-O datasets. On the YCB-V dataset, it is superior to all competitors, with a large margin over the second-best RGB method. 

![EPOSfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/EPOSfig2.png)

---

#### [7] BodiesatRest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data [PR20](https://arxiv.org/pdf/2004.01166.pdf) 

![BRT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/BRT.png)

People spend a substantial part of their lives at rest in bed. 3D human pose and shape estimation for this activity would have numerous beneficial applications, yet line-of-sight perception is complicated by occlusion from bedding. Pressure sensing mats are a promising alternative, but training data is challenging to collect at scale. We describe a physics-based method that simulates human bodies at rest in a bed with a pressure sensing mat, and present PressurePose, a synthetic dataset with 206K pressure images with 3D human poses and shapes. We also present PressureNet, a deep learning model that estimates human pose and shape given a pressure image and gender. PressureNet incorporates a pressure map reconstruction (PMR) network that models pressure image generation to promote consistency between estimated 3D body models and pressure image input. In our evaluations, PressureNet performed well with real data from participants in diverse poses, even though it had only been trained with synthetic data. When we ablated the PMR network, performance dropped substantially.

![BRfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/BRfig2.png)

---

#### [8] G2LNet: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features [PR20](https://arxiv.org/pdf/2003.11089.pdf) [Torch](https://github.com/DC1991/G2L_Net) 

![G2LNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/G2LNetT.png)

In this paper, we propose a novel real-time 6D object pose estimation framework, named G2L-Net. Our network operates on point clouds from RGB-D detection in a divide-and-conquer fashion. Specifically, our network consists of three steps. First, we extract the coarse object point cloud from the RGB-D image by 2D detection. Second, we feed the coarse object point cloud to a translation localization network to perform 3D segmentation and object translation prediction. Third, via the predicted segmentation and translation, we transfer the fine object point cloud into a local canonical coordinate, in which we train a rotation localization network to estimate initial object rotation. In the third step, we define point-wise embedding vector features to capture viewpoint-aware information. To calculate more accurate rotation, we adopt a rotation residual estimator to estimate the residual between initial rotation and ground truth, which can boost initial pose estimation performance. Our proposed G2L-Net is real-time despite the fact multiple steps are stacked via the proposed coarse-to-fine framework. Extensive experiments on two benchmark datasets show that G2L-Net achieves state-of-the-art performance in terms of both accuracy and speed.

![G2LNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/G2LNetfig2.png)

---

#### [9] Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS [PR20](<https://arxiv.org/pdf/2003.03972.pdf>)

![CVTT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/CVTT.png)

Estimating 3D poses of multiple humans in real-time is a classic but still challenging task in computer vision. Its major difficulty lies in the ambiguity in cross-view association of 2D poses and the huge state space when there are multiple people in multiple views. In this paper, we present a novel solution for multi-human 3D pose estimation from multiple calibrated camera views. It takes 2D poses in different camera coordinates as inputs and aims for the accurate 3D poses in the global coordinate. Unlike previous methods that associate 2D poses among all pairs of views from scratch at every frame, we exploit the temporal consistency in videos to match the 2D inputs with 3D poses directly in 3-space. More specifically, we propose to retain the 3D pose for each person and update them iteratively via the cross-view multi-human tracking. This novel formulation improves both accuracy and efficiency, as we demonstrated on widely-used public datasets. To further verify the scalability of our method, we propose a new large-scale multihuman dataset with 12 to 28 camera views. Without bells and whistles, our solution achieves 154 FPS on 12 cameras and 34 FPS on 28 cameras, indicating its ability to handle large-scale real-world applications. The proposed dataset will be released soon.

![CVTfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/CVTfig2.png)

---

#### [10] Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild [PRO20](https://arxiv.org/pdf/2004.01946.pdf) [Project](https://www.arielai.com/mesh_hands/) 

![WSMCHRT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/WSMCHRT.png)

We introduce a simple and effective network architecture for monocular 3D hand pose estimation consisting of an image encoder followed by a mesh convolutional decoder that is trained through a direct 3D hand mesh reconstruction loss. We train our network by gathering a large-scale dataset of hand action in YouTube videos and use it as a source of weak supervision. Our weakly-supervised mesh convolutions-based system largely outperforms state-of-the-art methods, even halving the errors on the in the wild benchmark.

![WSMCHRfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/WSMCHRfig1.png)

---

#### [11] Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation [PR20](https://arxiv.org/pdf/2004.02186.pdf) 

![LMVT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/LMVT.png)

We present a lightweight solution to recover 3D pose from multi-view images captured with spatially calibrated cameras. Building upon recent advances in interpretable representation learning, we exploit 3D geometry to fuse input images into a unified latent representation of pose, which is disentangled from camera view-points. This allows us to reason effectively about 3D pose across different views without using compute-intensive volumetric grids. Our architecture then conditions the learned representation on camera projection operators to produce accurate per-view 2d detections, that can be simply lifted to 3D via a differentiable Direct Linear Transform (DLT) layer. In order to do it efficiently, we propose a novel implementation of DLT that is orders of magnitude faster on GPU architectures than standard SVD-based triangulation methods. We evaluate our approach on two large-scale human pose datasets (H36M and Total Capture): our method outperforms or performs comparably to the state-of-the-art volumetric methods, while, unlike them, yielding real-time performance.

![LMVfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Pose%20Estimation/LMVfig2.png)

------

