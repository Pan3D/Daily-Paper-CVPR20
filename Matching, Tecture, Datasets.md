### Matching

#### RPMNet: Robust Point Matching using Learned Features  [PR20](https://arxiv.org/pdf/2003.13479.pdf)  [Code](https://github.com/yewzijian/RPMNet)  

![RPMNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/RPMNetT.png)

Iterative Closest Point (ICP) solves the rigid point cloud registration problem iteratively in two steps: (1) make hard assignments of spatially closest point correspondences, and then (2) find the least-squares rigid transformation. The hard assignments of closest point correspondences based on spatial distances are sensitive to the initial rigid transformation and noisy/outlier points, which often cause ICP to converge to wrong local minima. In this paper, we propose the RPM-Net -- a less sensitive to initialization and more robust deep learning-based approach for rigid point cloud registration. To this end, our network uses the differentiable Sinkhorn layer and annealing to get soft assignments of point correspondences from hybrid features learned from both spatial coordinates and local geometry. To further improve registration performance, we introduce a secondary network to predict optimal annealing parameters. Unlike some existing methods, our RPM-Net handles missing correspondences and point clouds with partial visibility. Experimental results show that our RPM-Net achieves state-of-the-art performance compared to existing non-deep learning and recent deep learning methods. 

![RPMNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/RPMNetfig2.png)

---

#### Learning Multiview 3D Point Cloud Registration [PR20](<https://arxiv.org/pdf/2001.05119.pdf>) [Code](<https://github.com/zgojcic/3D_multiview_reg>) 

![LMVT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/LMVT.png)

We present a novel, end-to-end learnable, multiview 3D point cloud registration algorithm. Registration of multiple scans typically follows a two-stage pipeline: the initial pairwise alignment and the globally consistent refinement. The former is often ambiguous due to the low overlap of neighboring point clouds, symmetries and repetitive scene parts. Therefore, the latter global refinement aims at establishing the cyclic consistency across multiple scans and helps in resolving the ambiguous cases. In this paper we propose, to the best of our knowledge, the first end-to-end algorithm for joint learning of both parts of this two-stage problem. Experimental evaluation on well accepted benchmark datasets shows that our approach outperforms the state-of-the-art by a significant margin, while being end-to-end trainable and computationally less costly. Moreover, we present detailed analysis and an ablation study that validate the novel components of our approach.

![LMVfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/LMVfig2.png)

---

#### D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features [PR20](<https://arxiv.org/pdf/2003.03164.pdf>) [TF](<https://github.com/XuyangBai/D3Feat>) 

![D3FeatT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/D3FeatT.png)

A successful point cloud registration often lies on robust establishment of sparse matches through discriminative 3D local features. Despite the fast evolution of learning-based 3D feature descriptors, little attention has been drawn to the learning of 3D feature detectors, even less for a joint learning of the two tasks. In this paper, we leverage a 3D fully convolutional network for 3D point clouds, and propose a novel and practical learning mechanism that densely predicts both a detection score and a description feature for each 3D point. In particular, we propose a keypoint selection strategy that overcomes the inherent density variations of 3D point clouds, and further propose a self-supervised detector loss guided by the on-the-fly feature matching results during training. Finally, our method achieves state-of-the-art results in both indoor and outdoor scenarios, evaluated on 3DMatch and KITTI datasets, and shows its strong generalization ability on the ETH dataset. Towards practical use, we show that by adopting a reliable feature detector, sampling a smaller number of features is sufficient to achieve accurate and fast point cloud alignment.In this work, we propose an end-to-end framework to learn local multi-view descriptors for 3D point clouds. To adopt a similar multi-view representation, existing studies use hand-crafted viewpoints for rendering in a preprocessing stage, which is detached from the subsequent descriptor learning stage. In our framework, we integrate the multi-view rendering into neural networks by using a differentiable renderer, which allows the viewpoints to be optimizable parameters for capturing more informative local context of interest points. To obtain discriminative descriptors, we also design a soft-view pooling module to attentively fuse convolutional features across views. Extensive experiments on existing 3D registration benchmarks show that our method outperforms existing local descriptors both quantitatively and qualitatively.

![D3Featfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/D3Featfig1.png)

---

### Texture

#### Learning to Transfer Texture from Clothing Images to 3D Humans [PR20](<https://virtualhumans.mpi-inf.mpg.de/papers/mir20pix2surf/mir20pix2surf.pdf>) [Code](https://virtualhumans.mpi-inf.mpg.de/pix2surf/) 

![LTTT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/LTTT.png)

In this paper, we present a simple yet effective method to automatically transfer textures of clothing images (front and back) to 3D garments worn on top SMPL, in real time. We first automatically compute training pairs of images with aligned 3D garments using a custom non-rigid 3D to 2D registration method, which is accurate but slow. Using these pairs, we learn a mapping from pixels to the 3D garment surface. Our idea is to learn dense correspondences from garment image silhouettes to a 2D-UV map of a 3D garment surface using shape information alone, completely ignoring texture, which allows us to generalize to the wide range of web images. Several experiments demonstrate that our model is more accurate than widely used baselines such as thin-plate-spline warping and image-to-image translation networks while being orders of magnitude faster. Our model opens the door for applications such as virtual-try on, and allows generation of 3D humans with varied textures which is necessary for learning.

![LTTfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/LTTfig3.png)

---

#### TheVirtualTailor: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style [PR20](<https://virtualhumans.mpi-inf.mpg.de/papers/patel20vtailor/vtailor.pdf>) [Code](<https://virtualhumans.mpi-inf.mpg.de/vtailor/>) 

![TVTT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/TVTT.png)

In this paper, we present TailorNet, a neural model which predicts clothing deformation in 3D as a function of three factors: pose, shape and style (garment geometry), while retaining wrinkle detail. This goes beyond prior models, which are either specific to one style and shape, or generalize to different shapes producing smooth results, despite being style specific. Our hypothesis is that (even non-linear) combinations of examples smooth out high frequency components such as fine-wrinkles, which makes learning the three factors jointly hard. At the heart of our technique is a decomposition of deformation into a high frequency and a low frequency component. While the low-frequency component is predicted from pose, shape and style parameters with an MLP, the high-frequency component is predicted with a mixture of shape-style specific pose models. The weights of the mixture are computed with a narrow bandwidth kernel to guarantee that only predictions with similar high-frequency patterns are combined. The style variation is obtained by computing, in a canonical pose, a subspace of deformation, which satisfies physical constraints such as inter-penetration, and draping on the body. TailorNet delivers 3D garments which retain the wrinkles from the physics based simulations (PBS) it is learned from, while running more than 1000 times faster. In contrast to classical PBS, TailorNet is easy to use and fully differentiable, which is crucial for computer vision and learning algorithms. Several experiments demonstrate TailorNet produces more realistic results than prior work, and even generates temporally coherent deformations on sequences of the AMASS dataset, despite being trained on static poses from a different dataset. To stimulate further research in this direction, we will make a dataset consisting of 55800 frames.

![TVTfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/TVTfig2.png)

---

#### Learning to Dress 3D People in Generative Clothing [PR20](<https://arxiv.org/pdf/1907.13615.pdf>) 

![LDT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/LDT.png)

Three-dimensional human body models are widely used in the analysis of human pose and motion. Existing models, however, are learned from minimally-clothed 3D scans and thus do not generalize to the complexity of dressed people in common images and videos. Additionally, current models lack the expressive power needed to represent the complex non-linear geometry of pose-dependent clothing shape. To address this, we learn a generative 3D mesh model of clothed people from 3D scans with varying pose and clothing. Specifically, we train a conditional Mesh-VAE-GAN to learn the clothing deformation from the SMPL body model, making clothing an additional term on SMPL. Our model is conditioned on both pose and clothing type, giving the ability to draw samples of clothing to dress different body shapes in a variety of styles and poses. To preserve wrinkle detail, our Mesh-VAE-GAN extends patchwise discriminators to 3D meshes. Our model, named CAPE, represents global shape and fine local structure, effectively extending the SMPL body model to clothing. To our knowledge, this is the first generative model that directly dresses 3D human body meshes and generalizes to different poses.

![LDfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/LDfig3.png)

---

#### Towards Photo-Realistic Virtual Try-On by Adaptively Generating $\leftrightarrow $ Preserving Image Content [PR20](<https://arxiv.org/pdf/2003.05863.pdf>) 

![TPRVT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/TPRVT.png)

Image visual try-on aims at transferring a target clothing image onto a reference person, and has become a hot topic in recent years. Prior arts usually focus on preserving the character of a clothing image (e.g. texture, logo, embroidery) when warping it to arbitrary human pose. However, it remains a big challenge to generate photo-realistic try-on images when large occlusions and human poses are presented in the reference person. To address this issue, we propose a novel visual try-on network, namely Adaptive Content Generating and Preserving Network (ACGPN). In particular, ACGPN first predicts semantic layout of the reference image that will be changed after try-on (e.g. long sleeve shirt→arm, arm→jacket), and then determines whether its image content needs to be generated or preserved according to the predicted semantic layout, leading to photo-realistic try-on and rich clothing details. ACGPN generally involves three major modules. First, a semantic layout generation module utilizes semantic segmentation of the reference image to progressively predict the desired semantic layout after try-on. Second, a clothes warping module warps clothing images according to the generated semantic layout, where a second-order difference constraint is introduced to stabilize the warping process during training. Third, an inpainting module for content fusion integrates all information (e.g. reference image, semantic layout, warped clothes) to adaptively produce each semantic part of human body. In comparison to the state-of-the-art methods, ACGPN can generate photo-realistic images with much better perceptual quality and richer fine-details.

![TPRVfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/TPRVfig2.png)

---

### Datasets

#### SPARE3D: A Dataset for Spatial Reasoning on Three-View Line Drawings [PR20](https://arxiv.org/pdf/2003.14034.pdf) [Project](https://ai4ce.github.io/SPARE3D/) 

![SPARE3DT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/SPARE3DT.png)

Spatial reasoning is an important component of human intelligence. We can imagine the shapes of 3D objects and reason about their spatial relations by merely looking at their three-view line drawings in 2D, with different levels of competence. Can deep networks be trained to perform spatial reasoning tasks? How can we measure their "spatial intelligence"? To answer these questions, we present the SPARE3D dataset. Based on cognitive science and psychometrics, SPARE3D contains three types of 2D-3D reasoning tasks on view consistency, camera pose, and shape generation, with increasing difficulty. We then design a method to automatically generate a large number of challenging questions with ground truth answers for each task. They are used to provide supervision for training our baseline models using state-of-the-art architectures like ResNet. Our experiments show that although convolutional networks have achieved superhuman performance in many visual learning tasks, their spatial reasoning performance in SPARE3D is almost equal to random guesses. We hope SPARE3D can stimulate new problem formulations and network designs for spatial reasoning to empower intelligent robots to operate effectively in the 3D world via 2D sensors.

![SPARE3Dfig234](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/SPARE3Dfig234.png)

---

#### Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS [PR20](<https://arxiv.org/pdf/2003.03972.pdf>) 

![CVTT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/CVTT.png)

Estimating 3D poses of multiple humans in real-time is a classic but still challenging task in computer vision. Its major difficulty lies in the ambiguity in cross-view association of 2D poses and the huge state space when there are multiple people in multiple views. In this paper, we present a novel solution for multi-human 3D pose estimation from multiple calibrated camera views. It takes 2D poses in different camera coordinates as inputs and aims for the accurate 3D poses in the global coordinate. Unlike previous methods that associate 2D poses among all pairs of views from scratch at every frame, we exploit the temporal consistency in videos to match the 2D inputs with 3D poses directly in 3-space. More specifically, we propose to retain the 3D pose for each person and update them iteratively via the cross-view multi-human tracking. This novel formulation improves both accuracy and efficiency, as we demonstrated on widely-used public datasets. To further verify the scalability of our method, we propose a new large-scale multi-human dataset with 12 to 28 camera views. Without bells and whistles, our solution achieves 154 FPS on 12 cameras and 34 FPS on 28 cameras, indicating its ability to handle large-scale real-world applications. The proposed dataset will be released soon.

![CVTfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/CVTfig2.png)

---

#### IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning [PR20](<https://arxiv.org/pdf/2003.02920.pdf>) [Data]() 

![IntrAT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/IntrAT.png)

Medicine is an important application area for deep learning models. Research in this field is a combination of medical expertise and data science knowledge. In this paper, instead of 2D medical images, we introduce an open-access 3D intracranial aneurysm dataset, IntrA, that makes the application of points-based and mesh-based classification and segmentation models available. Our dataset can be used to diagnose intracranial aneurysms and to extract the neck for a clipping operation in medicine and other areas of deep learning, such as normal estimation and surface reconstruction. We provide a large-scale benchmark of classification and part segmentation by testing state-of-the-art networks. We also discuss the performance of each method and demonstrate the challenges of our dataset.

![IntrAfig7](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Matching%2C%20Tecture%2C%20Datasets/IntrAfig7.png)

---

