### Generation/Reconstruction

---

##### [1] PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization [PR20](https://arxiv.org/pdf/2004.00452.pdf) [Project](https://shunsukesaito.github.io/PIFuHD/) 

![PIFuHDT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/PIFuHDT.png)

Recent advances in image-based 3D human shape estimation have been driven by the significant improvement in representation power afforded by deep neural networks. Although current approaches have demonstrated the potential in real world settings, they still fail to produce reconstructions with the level of detail often present in the input images. We argue that this limitation stems primarily form two conflicting requirements; accurate predictions require large context, but precise predictions require high resolution. Due to memory limitations in current hardware, previous approaches tend to take low resolution images as input to cover large spatial context, and produce less precise (or low resolution) 3D estimates as a result. We address this limitation by formulating a multi-level architecture that is end-to-end trainable. A coarse level observes the whole image at lower resolution and focuses on holistic reasoning. This provides context to an fine level which estimates highly detailed geometry by observing higher-resolution images. We demonstrate that our approach significantly outperforms existing state-of-the-art techniques on single image human shape reconstruction by fully leveraging 1k-resolution input images.

![PIFuHDfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/PIFuHDfig2.png)

------

##### [2] Local Implicit Grid Representations for 3D Scenes [PR20](https://arxiv.org/pdf/2003.08981.pdf) [Video](https://www.youtube.com/watch?v=XCyl1-vxfII&feature=youtu.be) 

![LIGRT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/LIGRT.png)

Shape priors learned from data are commonly used to reconstruct 3D objects from partial or noisy data. Yet no such shape priors are available for indoor scenes, since typical 3D autoencoders cannot handle their scale, complexity, or diversity. In this paper, we introduce Local Implicit Grid Representations, a new 3D shape representation designed for scalability and generality. The motivating idea is that most 3D surfaces share geometric details at some scale -- i.e., at a scale smaller than an entire object and larger than a small patch. We train an autoencoder to learn an embedding of local crops of 3D shapes at that size. Then, we use the decoder as a component in a shape optimization that solves for a set of latent codes on a regular grid of overlapping crops such that an interpolation of the decoded local shapes matches a partial or noisy observation. We demonstrate the value of this proposed approach for 3D surface reconstruction from sparse point observations, showing significantly better results than alternative approaches.

![LIGRfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/LIGRfig2.png)

------

##### [3] DeepCap: Monocular Human Performance Capture Using Weak Supervision [PR20](<https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/>) 

![DeepCapT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/DeepCapT.png)

Human performance capture is a highly important computer vision problem with many applications in movie production and virtual/augmented reality. Many previous performance capture approaches either required expensive multi-view setups or did not recover dense space-time coherent geometry with frame-to-frame correspondences. We propose a novel deep learning approach for monocular dense human performance capture. Our method is trained in a weakly supervised manner based on multi-view supervision completely removing the need for training data with 3D ground truth annotations. The network architecture is based on two separate networks that disentangle the task into a pose estimation and a non-rigid surface deformation step. Extensive qualitative and quantitative evaluations show that our approach outperforms the state of the art in terms of quality and robustness.

![DeepCapfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/DeepCapfig2.png)

------

##### [4] 3DSketchAwareSSC: 3D Sketch-aware Semantic Scene Completion via Semi-supervised Structure Prior [PR20](<https://arxiv.org/pdf/2003.14052.pdf>) [Project](<https://charlescxk.github.io/>) [Code](<https://github.com/charlesCXK/3D-SketchAware-SSC>) 

![3DSSSCT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/3DSSSCT.png)

The goal of the Semantic Scene Completion (SSC) task is to simultaneously predict a completed 3D voxel representation of volumetric occupancy and semantic labels of objects in the scene from a single-view observation. Since the computational cost generally increases explosively along with the growth of voxel resolution, most current state-of-the-arts have to tailor their framework into a low-resolution representation with the sacrifice of detail prediction. Thus, voxel resolution becomes one of the crucial difficulties that lead to the performance bottleneck.
In this paper, we propose to devise a new geometry-based strategy to embed depth information with low-resolution voxel representation, which could still be able to encode sufficient geometric information, e.g., room layout, object's sizes and shapes, to infer the invisible areas of the scene with well structure-preserving details. To this end, we first propose a novel 3D sketch-aware feature embedding to explicitly encode geometric information effectively and efficiently. With the 3D sketch in hand, we further devise a simple yet effective semantic scene completion framework that incorporates a light-weight 3D Sketch Hallucination module to guide the inference of occupancy and the semantic labels via a semi-supervised structure prior learning strategy. We demonstrate that our proposed geometric embedding works better than the depth feature learning from habitual SSC frameworks. Our final model surpasses state-of-the-arts consistently on three public benchmarks, which only requires 3D volumes of 60 x 36 x 60 resolution for both input and output.

![3DSSSCfig3](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/3DSSSCfig3.png)

---

##### [5] PQNet: A Generative Part Seq2Seq Network for 3D Shapes [PR20](<https://arxiv.org/pdf/1911.10949.pdf>) [Torch](<https://github.com/ChrisWu1997/PQ-NET>) [Account](<https://mp.weixin.qq.com/s/SMDzPAJCpgjELsKHeQvW-A>) 

![PQNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/PQNetT.png)

We introduce PQ-NET, a deep neural network which represents and generates 3D shapes via sequential part assembly. The input to our network is a 3D shape segmented into parts, where each part is first encoded into a feature representation using a part autoencoder. The core component of PQ-NET is a sequence-to-sequence or Seq2Seq autoencoder which encodes a sequence of part features into a latent vector of fixed size, and the decoder reconstructs the 3D shape, one part at a time, resulting in a sequential assembly. The latent space formed by the Seq2Seq encoder encodes both part structure and fine part geometry. The decoder can be adapted to perform several generative tasks including shape autoencoding, interpolation, novel shape generation, and single-view 3D reconstruction, where the generated shapes are all composed of meaningful parts.

![PQNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/PQNetfig2.png)

---

##### [6] Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion [PR20](<https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf>) [Code](https://virtualhumans.mpiinf.mpg.de/ifnets/)

![IFFST](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/IFFST.png)

While many works focus on 3D reconstruction from images, in this paper, we focus on 3D shape reconstruction and completion from a variety of 3D inputs, which are deficient in some respect: low and high resolution voxels, sparse and dense point clouds, complete or incomplete. Processing of such 3D inputs is an increasingly important problem as they are the output of 3D scanners, which are becoming more accessible, and are the intermediate output of 3D computer vision algorithms. Recently, learned implicit functions have shown great promise as they produce continuous reconstructions. However, we identified two limitations in reconstruction from 3D inputs: 1) details present in the input data are not retained, and 2) poor reconstruction of articulated humans. To solve this, we propose Implicit Feature Networks (IF-Nets), which deliver continuous outputs, can handle multiple topologies, and complete shapes for missing or sparse input data retaining the nice properties of recent learned implicit functions, but critically they can also retain detail when it is present in the input data, and can reconstruct articulated humans. Our work differs from prior work in two crucial aspects. First, instead of using a single vector to encode a 3D shape, we extract a learnable 3-dimensional multi-scale tensor of deep features, which is aligned with the original Euclidean space embedding the shape. Second, instead of classifying xyz point coordinates directly, we classify deep features extracted from the tensor at a continuous query point. We show that this forces our model to make decisions based on global and local shape structure, as opposed to point coordinates, which are arbitrary under Euclidean transformations. Experiments demonstrate that IF-Nets outperformprior work in 3D object reconstruction in ShapeNet, and obtain significantly more accurate 3D human reconstructions.

![IFFSfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/IFFSfig2.png)

---

##### [7] InPerfectShape: Certifiably Optimal 3D Shape Reconstruction from 2D Landmarks [PR20](https://arxiv.org/pdf/1911.11924.pdf)

![IPST](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/IPST.png)

We study the problem of 3D shape reconstruction from 2D landmarks extracted in a single image. We adopt the 3D deformable shape model and formulate the reconstruction as a joint optimization of the camera pose and the linear shape parameters. Our first contribution is to apply Lasserre's hierarchy of convex Sums-of-Squares (SOS) relaxations to solve the shape reconstruction problem and show that the SOS relaxation of minimum order 2 empirically solves the original non-convex problem exactly. Our second contribution is to exploit the structure of the polynomial in the objective function and find a reduced set of basis monomials for the SOS relaxation that significantly decreases the size of the resulting semidefinite program (SDP) without compromising its accuracy. These two contributions, to the best of our knowledge, lead to the first certifiably optimal solver for 3D shape reconstruction, that we name Shape*. Our third contribution is to add an outlier rejection layer to Shape* using a truncated least squares (TLS) robust cost function and leveraging graduated non-convexity to solve TLS without initialization. The result is a robust reconstruction algorithm, named Shape#, that tolerates a large amount of outlier measurements. We evaluate the performance of Shape* and Shape# in both simulated and real experiments, showing that Shape* outperforms local optimization and previous convex relaxation techniques, while Shape# achieves state-of-the-art performance and is robust against 70% outliers in the FG3DCar dataset.

![IPSfig1](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/IPSfig1.png)

---

##### [8] PFNet: Point Fractal Network for 3D Point Cloud Completion [PR20](<https://arxiv.org/pdf/2003.00410.pdf>) [Torch](<https://github.com/zztianzz/PF-Net-Point-Fractal-Network>) [Account](<https://mp.weixin.qq.com/s/0b1FeYv6DMj-rLysSi6lYQ>)

![PFNetT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/PFNetT.png)

In this paper, we propose a Point Fractal Network (PF-Net), a novel learning-based approach for precise and high-fidelity point cloud completion. Unlike existing point cloud completion networks, which generate the overall shape of the point cloud from the incomplete point cloud and always change existing points and encounter noise and geometrical loss, PF-Net preserves the spatial arrangements of the incomplete point cloud and can figure out the detailed geometrical structure of the missing region(s) in the prediction. To succeed at this task, PF-Net estimates the missing point cloud hierarchically by utilizing a feature-points-based multi-scale generating network. Further, we add up multi-stage completion loss and adversarial loss to generate more realistic missing region(s). The adversarial loss can better tackle multiple modes in the prediction. Our experiments demonstrate the effectiveness of our method for several challenging point cloud completion tasks.

![PFNetfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/PFNetfig2.png)

---

##### [9] Cascaded Refinement Network for Point Cloud Completion [PR20](<https://arxiv.org/pdf/2004.03327.pdf>) [Code](<https://github.com/xiaogangw/cascaded-point-completion>)

![CRNT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/CRNT.png)

Point clouds are often sparse and incomplete. Existing shape completion methods are incapable of generating details of objects or learning the complex point distributions. To this end, we propose a cascaded refinement network together with a coarse-to-fine strategy to synthesize the detailed object shapes. Considering the local details of partial input with the global shape information together, we can preserve the existing details in the incomplete point set and generate the missing parts with high fidelity. We also design a patch discriminator that guarantees every local area has the same pattern with the ground truth to learn the complicated point distribution. Quantitative and qualitative experiments on different datasets show that our method achieves superior results compared to existing state-of-the-art approaches on the 3D point cloud completion task.

![CRNfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/CRNfig2.png)

---

##### [10] Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions [PR20](<https://arxiv.org/pdf/2004.03967.pdf>) [Video](<https://www.youtube.com/watch?v=8D3HjYf6cYw&feature=youtu.be>) 

![LSSGT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/LSSGT.png)

Scene understanding has been of high interest in computer vision. It encompasses not only identifying objects in a scene, but also their relationships within the given context. With this goal, a recent line of works tackles 3D semantic segmentation and scene layout prediction. In our work we focus on scene graphs, a data structure that organizes the entities of a scene in a graph, where objects are nodes and their relationships modeled as edges. We leverage inference on scene graphs as a way to carry out 3D scene understanding, mapping objects and their relationships. In particular, we propose a learned method that regresses a scene graph from the point cloud of a scene. Our novel architecture is based on PointNet and Graph Convolutional Networks (GCN). In addition, we introduce 3DSSG, a semi-automatically generated dataset, that contains semantically rich scene graphs of 3D scenes. We show the application of our method in a domain-agnostic retrieval task, where graphs serve as an intermediate representation for 3D-3D and 2D-3D matching.

![LSSGfig4](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/LSSGfig4.png)

---

##### [11] ARCH: Animatable Reconstruction of Clothed Humans [PR20](https://arxiv.org/pdf/2004.04572.pdf) 

![ARCHT](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/ARCHT.png)

In this paper, we propose ARCH (Animatable Reconstruction of Clothed Humans), a novel end-to-end framework for accurate reconstruction of animation-ready 3D clothed humans from a monocular image. Existing approaches to digitize 3D humans struggle to handle pose variations and recover details. Also, they do not produce models that are animation ready. In contrast, ARCH is a learned pose-aware model that produces detailed 3D rigged full-body human avatars from a single unconstrained RGB image. A Semantic Space and a Semantic Deformation Field are created using a parametric 3D body estimator. They allow the transformation of 2D/3D clothed humans into a canonical space, reducing ambiguities in geometry caused by pose variations and occlusions in training data. Detailed surface geometry and appearance are learned using an implicit function representation with spatial local features. Furthermore, we propose additional per-pixel supervision on the 3D reconstruction using opacity-aware differentiable rendering. Our experiments indicate that ARCH increases the fidelity of the reconstructed humans. We obtain more than 50% lower reconstruction errors for standard metrics compared to state-of-the-art methods on public datasets. We also show numerous qualitative examples of animated, high-quality reconstructed avatars unseen in the literature so far.

![ARCHfig2](https://github.com/Pan3D/Daily-Paper-CVPR20/blob/master/Generation%20Reconstruction/ARCHfig2.png)

---

