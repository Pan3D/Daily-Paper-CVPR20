

### Keypoints, Layout, Depth, Surface

### Keypoints

##### [1] PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation [PR20](https://arxiv.org/abs/1911.04231) [Torch](<https://github.com/ethnhe/PVN3D>) [Video](<https://www.bilibili.com/video/av89408773/>) 

![PVN3D A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/PVN3D A.png)

In this work, we present a novel data-driven method for robust 6DoF object pose estimation from a single RGBD image. Unlike previous methods that directly regressing pose parameters, we tackle this challenging task with a keypoint-based approach. Specifically, we propose a deep Hough voting network to detect 3D keypoints of objects and then estimate the 6D pose parameters within a least-squares fitting manner. Our method is a natural extension of 2D-keypoint approaches that successfully work on RGB based 6DoF estimation. It allows us to fully utilize the geometric constraint of rigid objects with the extra depth information and is easy for a network to learn and optimize. Extensive experiments were conducted to demonstrate the effectiveness of 3D-keypoint detection in the 6D pose estimation task. Experimental results also show our method outperforms the state-of-the-art methods by large margins on several benchmarks.![PVN3D fig2](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/PVN3D fig2.png)

---

##### [2] KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations [PR20](<https://arxiv.org/pdf/2002.12687.pdf>)

![KeypointNet A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/KeypointNet A.png)

Detecting 3D objects keypoints is of great interest to the areas of both graphics and computer vision. There have been several 2D and 3D keypoint datasets aiming to address this problem in a data-driven way. These datasets, however, either lack scalability or bring ambiguity to the definition of keypoints. Therefore, we present KeypointNet: the first large-scale and diverse 3D keypoint dataset that contains 83,231 keypoints and 8,329 3D models from 16 object categories, by leveraging numerous human annotations. To handle the inconsistency between annotations from different people, we propose a novel method to aggregate these keypoints automatically, through minimization of a fidelity loss. Finally, ten state-of-the-art methods are benchmarked on our proposed dataset.![KeypointNet fig4](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/KeypointNet fig4.png)

------

### Layout

##### [3] Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction  for Indoor Scenes from a Single Image [PR20](<https://arxiv.org/pdf/2002.12212.pdf>) 

![Total3DUnderstanding A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/Total3DUnderstanding A.png)

Semantic reconstruction of indoor scenes refers to both scene understanding and object reconstruction. Existing works either address one part of this problem or focus on independent objects. In this paper, we bridge the gap between understanding and reconstruction, and propose an end-to-end solution to jointly reconstruct room layout, object bounding boxes and meshes from a single image. Instead of separately resolving scene understanding and object reconstruction, our method builds upon a holistic scene context and proposes a coarse-to-fine hierarchy with three components: 1. room layout with camera pose; 2. 3D object bounding boxes; 3. object meshes. We argue that understanding the context of each component can assist the task of parsing the others, which enables joint understanding and reconstruction. The experiments on the SUN RGB-D and Pix3D datasets demonstrate that our method consistently outperforms existing methods in indoor layout estimation, 3D object detection and mesh reconstruction.

![Total3DUnderstanding fig2](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/Total3DUnderstanding fig2.png)

---

##### [4] IntelligentHome3D: Automatic 3D-House Design from Linguistic Descriptions Only [PR20](<https://arxiv.org/pdf/2003.00397.pdf>) 

![IH A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/IH A.png)

Home design is a complex task that normally requires architects to finish with their professional skills and tools. It will be fascinating that if one can produce a house plan intuitively without knowing much knowledge about home design and experience of using complex designing tools, for example, via natural language. In this paper, we formulate it as a language conditioned visual content generation problem that is further divided into a floor plan generation and an interior texture (such as floor and wall) synthesis task. The only control signal of the generation process is the linguistic expression given by users that describe the house details. To this end, we propose a House Plan Generative Model (HPGM) that first translates the language input to a structural graph representation and then predicts the layout of rooms with a Graph Conditioned Layout Prediction Network (GC LPN) and generates the interior texture with a Language Conditioned Texture GAN (LCT-GAN). With some post-processing, the final product of this task is a 3D house model. To train and evaluate our model, we build the first Text-to-3D House Model dataset.

![IH fig2](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/IH fig2.png)

------

### Depth

##### [5] Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera [PR20](https://arxiv.org/pdf/2004.01294.pdf) 

![NVSDS A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/NVSDS A.png)

This paper presents a new method to synthesize an image from arbitrary views and times given a collection of images of a dynamic scene. A key challenge for the novel view synthesis arises from dynamic scene reconstruction where epipolar geometry does not apply to the local motion of dynamic contents. To address this challenge, we propose to combine the depth from single view (DSV) and the depth from multi-view stereo (DMV), where DSV is complete, i.e., a depth is assigned to every pixel, yet view-variant in its scale, while DMV is view-invariant yet incomplete. Our insight is that although its scale and quality are inconsistent with other views, the depth estimation from a single view can be used to reason about the globally coherent geometry of dynamic contents. We cast this problem as learning to correct the scale of DSV, and to refine each depth with locally consistent motions between views to form a coherent depth estimation. We integrate these tasks into a depth fusion network in a self-supervised fashion. Given the fused depth maps, we synthesize a photorealistic virtual view in a specific location and time with our deep blending network that completes the scene and renders the virtual view. We evaluate our method of depth estimation and view synthesis on diverse real-world dynamic scenes and show the outstanding performance over existing methods.![NVSDS fig3](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/NVSDS fig3.png)

----

##### [6] Towards Better Generalization: Joint Depth-Pose Learning without PoseNet [PR20](https://arxiv.org/pdf/2004.01314.pdf) [Torch](https://github.com/B1ueber2y/TrianFlow) [Pose Estimation](#Pose Estimation)

![TBG A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/TBG A.png)

In this work, we tackle the essential problem of scale inconsistency for self-supervised joint depth-pose learning. Most existing methods assume that a consistent scale of depth and pose can be learned across all input samples, which makes the learning problem harder, resulting in degraded performance and limited generalization in indoor environments and long-sequence visual odometry application. To address this issue, we propose a novel system that explicitly disentangles scale from the network estimation. Instead of relying on PoseNet architecture, our method recovers relative pose by directly solving fundamental matrix from dense optical flow correspondence and makes use of a two-view triangulation module to recover an up-to-scale 3D structure. Then, we align the scale of the depth prediction with the triangulated point cloud and use the transformed depth map for depth error computation and dense reprojection check. Our whole system can be jointly trained end-to-end. Extensive experiments show that our system not only reaches state-of-the-art performance on KITTI depth and flow estimation, but also significantly improves the generalization ability of existing self-supervised depth-pose learning methods under a variety of challenging scenarios, and achieves state-of-the-art results among self-supervised learning-based methods on KITTI Odometry and NYUv2 dataset. Furthermore, we present some interesting findings on the limitation of PoseNet-based relative pose estimation methods in terms of generalization ability. ![TBG fig2](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/TBG fig2.png)

------

### Surface

##### [7] Articulation-aware Canonical Surface Mapping [PR20](https://arxiv.org/pdf/2004.00614.pdf) [Torch](https://github.com/nileshkulkarni/acsm/) [Project](https://nileshkulkarni.github.io/acsm/) 

![ACSM A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/ACSM A.png)

We tackle the tasks of: 1) predicting a Canonical Surface Mapping (CSM) that indicates the mapping from 2D pixels to corresponding points on a canonical template shape, and 2) inferring the articulation and pose of the template corresponding to the input image. While previous approaches rely on keypoint supervision for learning, we present an approach that can learn without such annotations. Our key insight is that these tasks are geometrically related, and we can obtain supervisory signal via enforcing consistency among the predictions. We present results across a diverse set of animal object categories, showing that our method can learn articulation and CSM prediction from image collections using only foreground mask labels for training. We empirically show that allowing articulation helps learn more accurate CSM prediction, and that enforcing the consistency with predicted CSM is similarly critical for learning meaningful articulation.![ACSM fig45](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/ACSM fig45.png)

---

##### [8] Deep 3D Capture: Geometry and Reflectance from Sparse Multi-View Images [PR20](https://arxiv.org/pdf/2003.12642.pdf) 

![D3DC A](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/D3DC A.png)

We introduce a novel learning-based method to reconstruct the high-quality geometry and complex, spatially-varying BRDF of an arbitrary object from a sparse set of only six images captured by wide-baseline cameras under collocated point lighting. We first estimate per-view depth maps using a deep multi-view stereo network; these depth maps are used to coarsely align the different views. We propose a novel multi-view reflectance estimation network architecture that is trained to pool features from these coarsely aligned images and predict per-view spatially-varying diffuse albedo, surface normals, specular roughness and specular albedo. We do this by jointly optimizing the latent space of our multi-view reflectance network to minimize the photometric error between images rendered with our predictions and the input images. While previous state-of-the-art methods fail on such sparse acquisition setups, we demonstrate, via extensive experiments on synthetic and real data, that our method produces high-quality reconstructions that can be used to render photorealistic images.![D3DC fig2](https://github.com/Pan3D/Daily-Paper-CVPR20/tree/master/Keypoints%2C%20Layout%2C%20Depth%2C%20Surface/D3DC fig2.png)

------

### 
