### Detection

---

##### [1] FocalMix: Semi-Supervised Learning for 3D Medical Image Detection [PR20](https://arxiv.org/pdf/2003.09108.pdf) 

![FocalMixT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/FocalMixT.png)

Applying artificial intelligence techniques in medical imaging is one of the most promising areas in medicine. However, most of the recent success in this area highly relies on large amounts of carefully annotated data, whereas annotating medical images is a costly process. In this paper, we propose a novel method, called FocalMix, which, to the best of our knowledge, is the first to leverage recent advances in semi-supervised learning (SSL) for 3D medical image detection. We conducted extensive experiments on two widely used datasets for lung nodule detection, LUNA16 and NLST. Results show that our proposed SSL methods can achieve a substantial improvement of up to 17.3% over state-of-the-art supervised learning approaches with 400 unlabeled CT scans.

![FocalMixfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/FocalMixfig2.png)

---

##### [2] LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention [PR20](https://arxiv.org/pdf/2004.01389.pdf) [Code](https://github.com/yinjunbo/3DVID) 

![LOT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/LOT.png)

Existing LiDAR-based 3D object detectors usually focus on the single-frame detection, while ignoring the spatiotemporal information in consecutive point cloud frames. In this paper, we propose an end-to-end online 3D video object detector that operates on point cloud sequences. The proposed model comprises a spatial feature encoding component and a spatiotemporal feature aggregation component. In the former component, a novel Pillar Message Passing Network (PMPNet) is proposed to encode each discrete point cloud frame. It adaptively collects information for a pillar node from its neighbors by iterative message passing, which effectively enlarges the receptive field of the pillar feature. In the latter component, we propose an Attentive Spatiotemporal Transformer GRU (AST-GRU) to aggregate the spatiotemporal information, which enhances the conventional ConvGRU with an attentive memory gating mechanism. AST-GRU contains a Spatial Transformer Attention (STA) module and a Temporal Transformer Attention (TTA) module, which can emphasize the foreground objects and align the dynamic objects, respectively. Experimental results demonstrate that the proposed 3D video object detector achieves state-of-the-art performance on the large-scale nuScenes benchmark.

![LOfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/LOfig2.png)

---

##### [3] DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes [PR20](https://arxiv.org/pdf/2004.01170.pdf) 

![DOPST](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/DOPST.png)

We propose DOPS, a fast single-stage 3D object detection method for LIDAR data. Previous methods often make domain-specific design decisions, for example projecting points into a bird-eye view image in autonomous driving scenarios. In contrast, we propose a general-purpose method that works on both indoor and outdoor scenes. The core novelty of our method is a fast, single-pass architecture that both detects objects in 3D and estimates their shapes. 3D bounding box parameters are estimated in one pass for every point, aggregated through graph convolutions, and fed into a branch of the network that predicts latent codes representing the shape of each detected object. The latent shape space and shape decoder are learned on a synthetic dataset and then used as supervision for the end-to-end training of the 3D object detection pipeline. Thus our model is able to extract shapes without access to ground-truth shape information in the target dataset. During experiments, we find that our proposed method achieves state-of-the-art results by ~5% on object detection in ScanNet scenes, and it gets top results by 3.4% in the Waymo Open Dataset, while reproducing the shapes of detected cars.

![DOPSfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/DOPSfig2.png)

---

##### [4] PointGNN: Graph Neural Network for 3D Object Detection in a Point Cloud [PR20](<https://arxiv.org/pdf/2003.01251.pdf>)  [TF](<https://github.com/WeijingShi/Point-GNN>) [Graph]

![PointGNNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/PointGNNT.png)

In this paper, we propose a graph neural network to detect objects from a LiDAR point cloud. Towards this end, we encode the point cloud efficiently in a fixed radius near-neighbors graph. We design a graph neural network, named Point-GNN, to predict the category and shape of the object that each vertex in the graph belongs to. In Point-GNN, we propose an auto-registration mechanism to reduce translation variance, and also design a box merging and scoring operation to combine detections from multiple vertices accurately. Our experiments on the KITTI benchmark show the proposed approach achieves leading accuracy using the point cloud alone and can even surpass fusion-based algorithms. Our results demonstrate the potential of using the graph neural network as a new approach for 3D object detection.

![PointGNNfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/PointGNNfig2.png)

------

##### [5] MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships [PR20](<https://arxiv.org/pdf/2003.00504.pdf>) 

![MonoPairT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/MonoPairT.png)

Monocular 3D object detection is an essential component in autonomous driving while challenging to solve, especially for those occluded samples which are only partially visible. Most detectors consider each 3D object as an independent training target, inevitably resulting in a lack of useful information for occluded samples. To this end, we propose a novel method to improve the monocular 3D object detection by considering the relationship of paired samples. This allows us to encode spatial constraints for partially-occluded objects from their adjacent neighbors. Specifically, the proposed detector computes uncertainty-aware predictions for object locations and 3D distances for the adjacent object pairs, which are subsequently jointly optimized by nonlinear least squares. Finally, the one-stage uncertainty-aware prediction structure and the post-optimization module are dedicatedly integrated for ensuring the run-time efficiency. Experiments demonstrate that our method yields the best performance on KITTI 3D detection benchmark, by outperforming state-of-the-art competitors by wide margins, especially for the hard samples.

![MonoPairfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/MonoPairfig1.png)

------

##### [6] MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps [PR20](<https://arxiv.org/pdf/2003.06754.pdf>) [Code](<https://github.com/pxiangwu/MotionNet>) 

![MotionNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/MotionNetT.png)

The ability to reliably perceive the environmental states, particularly the existence of objects and their motion behavior, is crucial for autonomous driving. In this work, we propose an efficient deep model, called MotionNet, to jointly perform perception and motion prediction from 3D point clouds. MotionNet takes a sequence of LiDAR sweeps as input and outputs a bird’s eye view (BEV) map, which encodes the object category and motion information in each grid cell. The backbone of MotionNet is a novel spatiotemporal pyramid network, which extracts deep spatial and temporal features in a hierarchical fashion. To enforce the smoothness of predictions over both space and time, the training of MotionNet is further regularized with novel spatial and temporal consistency losses. Extensive experiments show that the proposed method overall outperforms the state-of-the-arts, including the latest scene-flow- and
3D-object-detection-based methods. This indicates the potential value of the proposed method serving as a backup to the bounding-box-based system, and providing complementary information to the motion planner in autonomous driving.

![MotionNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/MotionNetfig2.png)

------

##### [7] HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection [PR20](<https://arxiv.org/pdf/2003.00186.pdf>) [Account](https://mp.weixin.qq.com/s/prSkJIlVLdINBvNI8eUgvA) 

![HVNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/HVNetT.png)

We present Hybrid Voxel Network (HVNet), a novel one-stage unified network for point cloud based 3D object detection for autonomous driving. Recent studies show that 2D voxelization with per voxel PointNet style feature extractor leads to accurate and efficient detector for large 3D scenes. Since the size of the feature map determines the computation and memory cost, the size of the voxel becomes a parameter that is hard to balance. A smaller voxel size gives a better performance, especially for small objects, but a longer inference time. A larger voxel can cover the same area with a smaller feature map, but fails to capture intricate features and accurate location for smaller objects. We present a Hybrid Voxel network that solves this problem by fusing voxel feature encoder (VFE) of different scales at point-wise level and project into multiple pseudo-image feature maps. We further propose an attentive voxel feature encoding that outperforms plain VFE and a feature fusion pyramid network to aggregate multi-scale information at feature map level. Experiments on the KITTI benchmark show that a single HVNet achieves the best mAP among all existing methods with a real time inference speed of 31Hz.![HVNetfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/HVNetfig3.png)

---

##### [8] Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation [PR20](<https://arxiv.org/pdf/2004.03572.pdf>) [Torch](<https://github.com/zju3dv/disprcnn>) 

![DispRCNNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/DispRCNNT.png)

In this paper, we propose a novel system named Disp R-CNN for 3D object detection from stereo images. Many recent works solve this problem by first recovering a point cloud with disparity estimation and then apply a 3D detector. The disparity map is computed for the entire image, which is costly and fails to leverage category-specific prior. In contrast, we design an instance disparity estimation network (iDispNet) that predicts disparity only for pixels on objects of interest and learns a category-specific shape prior for more accurate disparity estimation. To address the challenge from scarcity of disparity annotation in training, we propose to use a statistical shape model to generate dense disparity pseudo-ground-truth without the need of LiDAR point clouds, which makes our system more widely applicable. Experiments on the KITTI dataset show that, even when LiDAR ground-truth is not available at training time, Disp R-CNN achieves competitive performance and outperforms previous state-of-the-art methods by 20% in terms of average precision.

![DispRCNNfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/DispRCNNfig2.png)

---

##### [9] End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection [PR20](<https://arxiv.org/pdf/2004.03080.pdf>) [Code](<https://github.com/mileyan/pseudo-LiDAR_e2e>) 

![E2EPLT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/E2EPLT.png)

Reliable and accurate 3D object detection is a necessity for safe autonomous driving. Although LiDAR sensors can provide accurate 3D point cloud estimates of the environment, they are also prohibitively expensive for many settings. Recently, the introduction of pseudo-LiDAR (PL) has led to a drastic reduction in the accuracy gap between methods based on LiDAR sensors and those based on cheap stereo cameras. PL combines state-of-the-art deep neural networks for 3D depth estimation with those for 3D object detection by converting 2D depth map outputs to 3D point cloud inputs. However, so far these two networks have to be trained separately. In this paper, we introduce a new framework based on differentiable Change of Representation (CoR) modules that allow the entire PL pipeline to be trained end-to-end. The resulting framework is compatible with most state-of-the-art networks for both tasks and in combination with PointRCNN improves over PL consistently across all benchmarks -- yielding the highest entry on the KITTI image-based 3D object detection leaderboard at the time of submission.

![E2EPLfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Detection/E2EPLfig3.png)

------
