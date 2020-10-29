## 2018-12

Fortunately, IPMI19 ddl is extended to Dec 13. Then I'm preparing one another MIC journal paper.

### Study
* Interpretable Machine Learning book ([link](https://christophm.github.io/interpretable-ml-book/index.html))

### Reading
* [x] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ([arXiv](https://arxiv.org/abs/1810.04805))
* MIC
  * [x] Risk Stratification of Lung Nodules Using 3D CNN-Based Multi-task Learning ([IPMI17](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_20)): too many details missing in the paper, e.g., how large is the input, detailed network architecture, MTL solution. Note the MTL is not end-to-end. It is not convincing in a cross-validation setting.
  * [x] Joint Learning for Pulmonary Nodule Segmentation, Attributes and Malignancy Prediction ([ISBI18](https://arxiv.org/abs/1802.03584)): it uses different accuracy defination. Very engineering-orient methodology, however not detailed and convincing.
  * [x] In Silico Labeling: Predicting Fluorescent Labels in Unlabeled Images ([Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)30364-7)): a very inspiring work. However, Cell paper seems very biology-oriented (at least this one). Some terminology is too complicated for me; thus sadly, I do not understand all the biomedcine translational values :( . Study hard! 
  * [x] fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ([arXiv](https://arxiv.org/abs/1811.08839)) ([code](https://github.com/facebookresearch/fastMRI)): "self-contained" paper, nice dataset, the community needs more like this. However, it is not so "self-contained" indeed, more background is needed to carry out brilliant research. It is a good start.
  * [ ] Transferable Multi-model Ensemble for Benign-Malignant Lung Nodule Classification on Chest CT ([MICCAI18](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_75))
  * [ ] Semi-Supervised Multi-Task Learning for Lung Cancer Diagnosis ([arXiv](https://arxiv.org/pdf/1802.06181.pdf))
  * [ ] Deep Active Self-paced Learning for Accurate Pulmonary Nodule Segmentation ([MICCAI18](https://www.researchgate.net/publication/327629896_Deep_Active_Self-paced_Learning_for_Accurate_Pulmonary_Nodule_Segmentation))
  * [ ] CT-Realistic Lung Nodule Simulation from 3D Conditional Generative Adversarial Networks for Robust Lung Segmentation ([MICCAI18](https://arxiv.org/abs/1806.04051))
  * [ ] Automated Pulmonary Nodule Detection: High Sensitivity with Few Candidates ([MICCAI18](https://www.researchgate.net/publication/327629744_Automated_Pulmonary_Nodule_Detection_High_Sensitivity_with_Few_Candidates_21st_International_Conference_Granada_Spain_September_16-20_2018_Proceedings_Part_II))
  * [ ] Discriminative Localization in CNNs for Weakly-Supervised Segmentation of Pulmonary Nodules ([MICCAI17](https://arxiv.org/abs/1707.01086))
  * [ ] Curriculum Adaptive Sampling for Extreme Data Imbalance ([MICCAI17](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_73)) ([code](https://github.com/taki0112/CASED-Tensorflow))
  * [ ] S4ND: Single-Shot Single-Scale Lung Nodule Detection ([MICCAI18](https://arxiv.org/abs/1805.02279))
  * [ ] DeepEM: Deep 3D ConvNets with EM for Weakly Supervised Pulmonary Nodule Detection ([MICCAI18](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_90))
  
## 2018-11
Very busy on preparing papers for CVPR19, then IPMI19.

## 2018-10
Main focus: preparing CVPR (and IPMI) [ddl](https://creedai.github.io/ai-deadlines/)

### Study
* [x] 一堂課讓你認識肺癌（Basic Concepts of Lung Cancer: Diagnosis and Treatment）([Coursera](https://www.coursera.org/learn/lung-cancer/))
* [x] Computational Neuroscience 计算神经科学 ([Xuetangx](http://www.xuetangx.com/courses/course-v1:NTHU+MOOC_04+sp/courseware/def9437b3df2456e88dd2e7fa0bb227a/))
  * [x] week3 - Signal propagation in neurons
  * [x] week4 - Neural Network simulators

### Reading
* 3D vision is still the main reading
  * [x] FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation ([CVPR2018](https://arxiv.org/abs/1712.07262))
  * [x] PU-Net: Point Cloud Upsampling Network ([CVPR2018](https://arxiv.org/abs/1801.06761))
  * [x] Pointwise Convolutional Neural Networks ([CVPR2018](https://arxiv.org/abs/1712.05245))
  * [x] 3D Graph Neural Networks for RGBD Semantic Segmentation ([ICCV2017](http://www.cs.toronto.edu/~rjliao/papers/iccv_2017_3DGNN.pdf))
  * [x] Recurrent Slice Networks for 3D Segmentation of Point Clouds ([CVPR2018](https://arxiv.org/abs/1802.04402))
  * [x] Learning Representations and Generative Models for 3D Point Clouds ([ICML2018](https://arxiv.org/abs/1707.02392)): apart from the GAN for point cloud, the metric to measure two point clouds is also useful (togethor with its [code](https://github.com/optas/latent_3d_points)). A paper solid to win (8-page supplementary experiments).
  * [x] PointGrid: A Deep Network for 3D Shape Understanding ([CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf)): a simple yet effective and efficient solution for point cloud. I do like this paper (though not very well-wrtien). However, it seems a purified methodology version of VoxelNet (also CVPR2018). No cross citation between these two papers.
  * [x] VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection ([CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3333.pdf)): a "PointGrid" (CVPR2018 as well) on kitti, better writen. No cross citation between these two papers.
  * [x] PointGrow: Autoregressively Learned Point Cloud Generation with Self-Attention ([OpenReview](https://openreview.net/forum?id=r1eWW2RqFX), ICLR2019 under review): not very clear writen. Some evaluation is mssing better metrics. Some baseline is missing. I guess a 60% probability to accept. Maybe writen by an intern of authors of DGCNN.
* Group Equivariance
  * [ ] Group Equivariant Convolutional Networks ([ICML2016](https://arxiv.org/abs/1602.07576)): as a premilary for other papers on this topic.
  * [ ] Learning Steerable Filters for Rotation Equivariant CNNs ([CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Weiler_Learning_Steerable_Filters_CVPR_2018_paper.pdf))
  * [ ] 3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data ([arXiv](https://arxiv.org/abs/1807.02547))
* Attention / Graph
  * [x] Hyperbolic Attention Networks ([OpenReview](https://openreview.net/forum?id=rJxHsjRqFQ), [arXiv](https://arxiv.org/abs/1805.09786), ICLR2019 under review): a paper I do not fully understand due to so much missing knowledge in Hyperbolic geometry. A Hyperblic space embedding seems appealing; However, it seems that this paper does not well explained why it is suitable for attention instead of general neural network representation. Worth more exploration.
  * [ ] Relational Graph Attention Networks ([OpenReview](https://openreview.net/forum?id=Bklzkh0qFm), ICLR2019 under review)
  * [ ] Hierarchical Graph Representation Learning with Differentiable Pooling ([arXiv](https://arxiv.org/abs/1806.08804))
  * [ ] Learning Visual Question Answering by Bootstrapping Hard Attention ([ECCV2018](https://arxiv.org/abs/1808.00300))
* [ ] A New Angle on L2 Regularization ([blog](https://thomas-tanay.github.io/post--L2-regularization/))
* [x] Efficient Annotation of Segmentation Datasets with PolygonRNN++ ([CVPR2018](http://www.cs.toronto.edu/polyrnn/)): very interesting application of existing algorithms (segmentation + RL + GNN), but some details are missing in this conference paper (maybe better in its journal version?). Many engineering details.
* [x] Taskonomy: Disentangling Task Transfer Learning ([CVPR2018 best](http://taskonomy.stanford.edu/)): hard to understand.
* [x] A Low Power, Fully Event-Based Gesture Recognition System ([CVPR2017](https://ieeexplore.ieee.org/document/8100264))
* [x] Hand PointNet: 3D Hand Pose Estimation using Point Sets ([CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ge_Hand_PointNet_3D_CVPR_2018_paper.pdf)): not my area actually. A PointNet application for hand pose regression.
* [x] Visible Machine Learning for Biomedicine ([Cell](https://linkinghub.elsevier.com/retrieve/pii/S0092-8674(18)30719-0) Commentary)

### 2018-09

Our paper `3D Deep Learning from CT Scans Predicts Tumor Invasiveness of Subcentimeter Pulmonary Adenocarcinomas` is accepted by  *[Cancer Research](http://cancerres.aacrjournals.org/content/early/2018/10/02/0008-5472.CAN-18-0696)* (DOI: 10.1158/0008-5472.CAN-18-0696).

### Reading
* 3D Vision
  * [ ] Spherical CNNs ([ICLR2018 best](https://openreview.net/forum?id=Hkbd5xZRb)): this paper is too complicated for me to understand :(
  * [x] Spherical convolutions and their application in molecular modelling ([NIPS2017](https://papers.nips.cc/paper/6935-spherical-convolutions-and-their-application-in-molecular-modelling)): difficult to read for non-native Engish. A good illustration for cubed-sphere grid ([link](http://acmg.seas.harvard.edu/geos/cubed_sphere/CubeSphere_step-by-step.html)).
  * [x] Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models ([ICCV2017](https://arxiv.org/abs/1704.01222))
  * [x] SO-Net: Self-Organizing Network for Point Cloud Analysis ([CVPR2018](https://arxiv.org/abs/1803.04249)): basically a PointNet++ with "SOM" clustering.
  * [x] SPLATNet: Sparse Lattice Networks for Point Cloud Processing ([CVPR2018 oral](https://arxiv.org/abs/1802.08275)): it uses differentiable projection to regular grids (permutohedral lattices), together with sparse convolution for efficiency. However, I have not understood its advantage over set-based networks (e.g., PointNet++). Partial reasons are that I have not understood the advantage of bilateral convolution layer (BCL).
  * [x] Neural 3D Mesh Renderer ([CVPR2018](https://arxiv.org/abs/1711.07566)) ([project page](http://hiroharu-kato.com/projects_en/neural_renderer.html))
    - *NB*: very fancy, very useful, but I have not fully understood the graphics-heavy work. I will renew my understanding further. tl;dr: 3 kinds of parameters can be optimized. `vertices`: [n_vertices, 3 (XYZ)], `textures` [n_faces, texture_size, texture_size, texture_size, 3 (RGB)], and `camera_position` [3]. Besides, `faces` [n_faces, 3 (triangle)] indicates the link of vertices (3 vs make a face), which makes the mesh can be processed like graph (it's graph indeed). `faces` seems can not be diff. The paper uses a Straight-Through Estimator to provide the gradients (for the vertices only, not sure at present; the others should have gradients naturally). 
  * [x] Generating 3D Adversarial Point Clouds ([arXiv](https://arxiv.org/abs/1809.07016)): poorly written.
  * [x] Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling ([arXiv](https://arxiv.org/abs/1712.06760)): not very insightful. Trivial modification uses too much language. Limited empirical improvements. However, learning visible (point) kernel is a good idea for interpretability in deep pc learning (which needs more exploration further).
  * [x] Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images ([ECCV2018](https://arxiv.org/abs/1804.01654)): though appealing, it needs 3D supervision, which is very different from N3MR.
* [x] Self-Attention Generative Adversarial Networks ([arXiv](https://arxiv.org/abs/1805.08318)): simple yet effective.
* [x] Spectral Normalization Explained ([paper: ICLR2018](https://openreview.net/forum?id=B1QRgziT-)) ([blog](https://christiancosgrove.com/blog/2018/01/04/spectral-normalization-explained.html)): greatly explained. SN means a $W/\sigma(W)$, $\sigma(W)$ denotes the max eigen vector of $W$. Then it provides a simple way to lower the computation.
* [x] Generative adversarial interpolative autoencoding: adversarial training on latent space interpolations encourage convex latent distributions ([arXiv](https://arxiv.org/abs/1807.06650)): an interesting paper, though intuitive.
* [x] Analyzing Inverse Problems with Invertible Neural Networks ([arXiv](https://arxiv.org/abs/1808.04730)): poorly written, hard to read.
* [x] Image Transformer ([ICML2018](https://arxiv.org/abs/1802.05751)): self-attention application to autoregressive models.
* [x] Self-Attention with Relative Position Representations ([NAACL2018](https://arxiv.org/abs/1803.02155)): a short paper, but provides good insight: instead of using pre-defined absolute postion-encoding, it uses learnable relative position embeddings. 
* [x] Universal Transformers ([arXiv](https://arxiv.org/abs/1807.03819)): just a simple modification: add recurrence in Transformers (i.e. sharing weights for multiple layers), plus a trivial ACT (just like my setting in the code ...)
* [x] A radiomics approach to assess tumour-infiltrating CD8 cells and response to anti-PD-1 or anti-PD-L1 immunotherapy: an imaging biomarker, retrospective multicohort study ([LANCET Oncology](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(18)30413-3/fulltext)): excellent angle on the usage of radiomics, though methodology is simple, the study is very meaningful and promising. 
* [x] Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples ([ICML2018 best paper](https://arxiv.org/abs/1802.00420))
  - *NB*: a brilliant conference paper like a journal paper (a research article, a review and a comment paper). Well-writen, comprehensive, well-performing. Very insightful. All the 8 pages are very worth reading. However, its journal version could be better (if it exists further), since some techniques proposed seem not well suited in its Case Study section, e.g., the Reparameterization for solving Vanisihing & Exploding Gradients seems not to appear; instead, it was solved by BPDA. Besides, I don't understand why LID appears in the "Gradient Shattering" section, let alone that LID was not circumbented by the 3 main attack techniques proposed. Overall, the paper developped a good story about 3 shields (Shattered Gradients, Stochastic Gradients and Exploding & Vanishing Gradients) and 3 swords (Backward Pass Differentiable Approximation BPDA, Expectation over Transformation EOT and Reparameterization), while one should be very cafeful that the shields & swords are not the whole of the paper. The comments in the discussion are also valuable for future studies.
* [x] Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning ([ICLR2016 -> T-PAMI](https://arxiv.org/abs/1704.03976)):
  - *NB*: good empirical results, elegant solutions. 3 Hyper-parameters: `eplison` (default: 8.0, norm length for (virtual) adversarial training), `num_power_iterations` or `K` (default: 1, the number of power iterations) and `xi` (default: 1e-6, small constant for finite difference). A clean PyTorch implementation [here](https://github.com/lyakaap/VAT-pytorch) but seems with wrong default hyper-p. For the VAT loss only, it needs `K+2` forwards, and `K+1` backwards (my calc is sightly different from the paper).

## 2018-08
My Bayesian month :)

### Reading
* Variantional Inference and Discrete Distribution
  * [x] A Tutorial on Variational Bayesian Inference ([pdf](http://www.orchid.ac.uk/eprints/40/1/fox_vbtut.pdf))
    - *NB*: very clear description on mean field interation, but seems non-trivial on VMP framework. Mean field is something using independant distribution to approximate the whole distribution.
  * [x] Tutorial on Variational Autoencoders ([arXiv](https://arxiv.org/abs/1606.05908))
  * [x] Categorical Reparameterization with Gumbel-Softmax ([ICLR2017](https://openreview.net/forum?id=rkE3y85ee))
  * [x] Learning Latent Permutations with Gumbel-Sinkhorn Networks ([ICLR2018](https://arxiv.org/abs/1802.08665)) 
  * [x] The Humble Gumbel Distribution ([blog](http://amid.fish/humble-gumbel))
* Generative Flow
  * [x] i-RevNet: Deep Invertible Networks ([ICLR2018](https://openreview.net/forum?id=HJsjkMb0Z)): a very interesting paper with numbers of potential applications; but it's not that novel at this time indeed, e.g., highly related to RealNVP and NICE.
  * [x] Glow: Generative Flow with Invertible 1x1 Convolutions ([arXiv](https://arxiv.org/abs/1807.03039)): introduce a trick Conv1x1 (based on Real NVP, Glow:RealNVP::DCGAN:GAN).
  * [x] Density estimation using Real NVP ([ICLR2017](https://arxiv.org/abs/1605.08803)): a **fantastic** paper.
  * [x] Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in Generative Models ([AAAI2018](https://arxiv.org/abs/1705.08868))
  * [x] Normalizing Flows Tutorial ([Part 1](https://blog.evjang.com/2018/01/nf1.html)) ([Part 2](https://blog.evjang.com/2018/01/nf2.html))
  * [x] Improving Variational Inference with Inverse Autoregressive Flow ([arXiv](https://arxiv.org/abs/1606.04934), plus a good [blog](http://bjlkeng.github.io/posts/variational-autoencoders-with-inverse-autoregressive-flows/))
  * [x] NICE: Non-linear Independent Components Estimation ([ICLR2015](https://arxiv.org/abs/1410.8516))
* [x] The Building Blocks of Interpretability ([Distill](https://distill.pub/2018/building-blocks/))
* [x] Sampling Generative Networks ([NIPS2016](https://arxiv.org/abs/1609.04468)): a good paper with bad writing.
* [x] Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer ([arXiv](https://arxiv.org/abs/1807.07543))
* [x] Instance Noise: A trick for stabilising GAN training ([blog](https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/))


## 2018-07
A busy month for reproducing [DeepLabv3+](https://github.com/duducheng/deeplabv3p_gluon) and NIPS rebuttal. 

### Reading
* [x] Learning Deep Matrix Representations ([arXiv](https://arxiv.org/abs/1703.01454))
* [x] Graph Memory Networks for Molecular Activity Prediction ([arXiv](https://arxiv.org/abs/1801.02622))
* [x] Weighted Transformer Network for Machine Translation ([OpenReview](https://openreview.net/forum?id=SkYMnLxRW))

## 2018-06
### Study Plan
* [ ] Introduction to Biomedical Imaging 生物医学成像学导论 ([MIC](topics/mic.md))
  * [x] Module 2 CT
  * [ ] Module 3 Ultrasounds
  * [ ] Module 4 MRI
  * [ ] Module 5 PET
* [x] Computational Neuroscience 计算神经科学 ([MIC](topics/mic.md))
  * [x] week1 - Basic neuronal models
  * [x] week2 - Synapse and channel dynamics

### Reading
* [x] A mixed-scale dense convolutional neural network for image analysis ([PNAS](http://www.pnas.org/content/115/2/254))
* [ ] Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation ([MICCAI2017](https://arxiv.org/abs/1706.04737))
* [ ] In Silico Labeling: Predicting Fluorescent Labels in Unlabeled Images ([Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)30364-7))
* [ ] A Tutorial on Variational Bayesian Inference ([pdf](http://www.orchid.ac.uk/eprints/40/1/fox_vbtut.pdf))
* [ ] Tutorial on Variational Autoencoders ([arXiv](https://arxiv.org/abs/1606.05908))
* [ ] World Models ([website](https://worldmodels.github.io/)) ([arXiv](https://arxiv.org/abs/1803.10122))
* [ ] The Building Blocks of Interpretability ([Distill](https://distill.pub/2018/building-blocks/))
* [x] Using Artiﬁcial Intelligence to Augment Human Intelligence ([Distill](https://distill.pub/2017/aia/))
* [ ] Memory-Efficient Implementation of DenseNets ([arXiv](https://arxiv.org/abs/1707.06990))
* [x] A Comparison of MCC and CEN Error Measures in Multi-Class Prediction ([PLOS ONE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3414515/))
* [x] Fully Convolutional Networks for Semantic Segmentation ([CVPR2015](https://arxiv.org/abs/1411.4038))
* [x] Pyramid Scene Parsing Network ([CVPR2017](https://arxiv.org/abs/1411.4038))
* [x] Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (a.k.a DeepLab v3+) ([arXiv](https://arxiv.org/abs/1802.02611))
* [x] Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks (a.k.a ODIN) ([ICLR2018](https://arxiv.org/abs/1706.02690))
* [x] Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery ([IPMI2017](https://arxiv.org/abs/1703.05921))
* [x] Anomaly Detection using One-Class Neural Networks ([KDD2018](https://arxiv.org/abs/1802.06360))
* [x] Thoughts on "mixup: Data-Dependent Data Augmentation" ([blog](http://www.inference.vc/mixup-data-dependent-data-augmentation/))
* 3D Vision & Point clouds
  * [x] RGBD Datasets: Past, Present and Future ([arXiv](https://arxiv.org/abs/1604.00999))
  * [x] 3D ShapeNets: A Deep Representation for Volumetric Shapes (the ModelNet dataset paper) ([CVPR2015](https://arxiv.org/abs/1406.5670))
* Few Shot
  * [x] Learning to Compare: Relation Network for Few-Shot Learning ([CVPR2018](https://arxiv.org/abs/1711.06025))
  * [x] Low-shot learning with large-scale diffusion ([CVPR2018](https://arxiv.org/abs/1706.02332))
  * [x] Few-Shot Image Recognition by Predicting Parameters from Activations ([CVPR2018](https://arxiv.org/abs/1706.03466))


## 2018-05
Still a very busy month for preparing papers.

### Reading
* EGFR
  * [x] Somatic mutations drive distinct imaging phenotypes in lung cancer ([Cancer Research](http://cancerres.aacrjournals.org/content/early/2017/05/31/0008-5472.CAN-17-0122))
  * [x] Defining a Radiomic Response Phenotype: A Pilot Study using targeted therapy in NSCLC ([Scientific Reports](https://www.nature.com/articles/srep33860))
  * [x] Non–Small Cell Lung Cancer Radiogenomics Map Identifies Relationships between Molecular and Imaging Phenotypes with Prognostic Implications ([Radiology](https://pubs.rsna.org/doi/10.1148/radiol.2017161845?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed))
  * [x] Radiomic Features Are Associated With EGFR Mutation Status in Lung Adenocarcinomas ([Clinical Lung Cancer](https://www.sciencedirect.com/science/article/pii/S1525730416300055?via%3Dihub))
* [ ] World Models ([website](https://worldmodels.github.io/)) ([arXiv](https://arxiv.org/abs/1803.10122))
* [ ] The Building Blocks of Interpretability ([Distill](https://distill.pub/2018/building-blocks/))
* [ ] Using Artiﬁcial Intelligence to Augment Human Intelligence ([Distill](https://distill.pub/2017/aia/))
* Dynamic Graph CNN for Learning on Point Clouds ([arXiv](https://arxiv.org/abs/1801.07829))

## 2018-04
A very busy month for preparing papers.

### Reading
* Interpretability
  * [x] Learning with Rejection ([page](https://cs.nyu.edu/~mohri/pub/rej.pdf))
  * [x] Predict Responsibly: Increasing Fairness by Learning To Defer ([arXiv](https://arxiv.org/abs/1711.06664))  
* PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space ([NIPS](https://arxiv.org/abs/1706.02413))
* PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation ([CVPR](https://arxiv.org/abs/1612.00593))

## 2018-03
### Study Plan
* [ ] lunglab-keras ([MIC](topics/mic.md))
* [x] Pointer Network in PyTorch and plus ([AML](topics/advanced_ml.md))
* [ ] Try radiomics and genomics ([MIC](topics/mic.md))

### Reading
* MIC
  * [x] Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs ([JAMA](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45732.pdf))
  * [x] Dermatologist-level classification of skin cancer with deep neural networks ([Nature](https://www.nature.com/articles/nature21056))
  * [x] Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning ([Cell](http://www.cell.com/cell/abstract/S0092-8674(18)30154-5))
  * [x] Scalable and accurate deep learning for electronic health records ([arXiv](https://arxiv.org/abs/1801.07860))
  * [x] CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning ([arXiv](https://arxiv.org/abs/1711.05225))
  * [x] Automated Pulmonary Nodule Detection via 3D ConvNets with Online Sample Filtering and Hybrid-Loss Residual Learning ([arXiv](https://arxiv.org/abs/1708.03867))
  * [x] DeepLung: 3D Deep Convolutional Nets for Automated Pulmonary Nodule Detection and Classification ([arXiv](https://arxiv.org/abs/1709.05538))
  * [x] A Survey on Deep Learning in Medical Image Analysis ([arXiv](https://arxiv.org/abs/1702.05747))
  * [x] Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach ([Nature Comm](https://www.nature.com/articles/ncomms5006))
  * [x] Computational Radiomics System to Decode the Radiographic Phenotype ([Cancer Research](http://cancerres.aacrjournals.org/content/77/21/e104.full-text.pdf))
  * [ ] Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation ([arXiv](https://arxiv.org/abs/1706.04737))
* MIL
  * [x] Graph Attention Networks ([arXiv](https://arxiv.org/abs/1710.10903))
  * [x] Deep learning of feature representation with multiple instance learning for medical image analysis ([ICASSP](http://ieeexplore.ieee.org/document/6853873/)): bad writing, trival idea, non-sense to read.
  * [x] Deep Multi-instance Networks with Sparse Label Assignment for Whole Mammogram Classification ([arXiv](https://arxiv.org/abs/1612.05968)) ([code](https://github.com/wentaozhu/deep-mil-for-whole-mammogram-classification))
  * [x] Multi-Instance Deep Learning: Discover Discriminative Local Anatomies for Bodypart Recognition: trival writing, same idea as DSB2017 THU solution.
  * [x] Learning from Experts: Developing Transferable Deep Features for Patient-Level Lung Cancer Prediction ([MICCAI](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_15)): very very bad writing, hard to find the details of the model / training / data. 
  * [x] Attention-based Deep Multiple Instance Learning ([ICML2018](https://arxiv.org/abs/1802.04712): !)
  * [x] Attention Solves Your TSP ([arXiv](https://arxiv.org/abs/1803.08475))
  * [ ] Multiple-Instance Learning for Medical Image and Video Analysis ([IEEE](http://ieeexplore.ieee.org/document/7812612/))
  * [ ] Revisiting Multiple Instance Neural Networks ([arXiv](https://arxiv.org/abs/1610.02501))
* [x] An introduction to ROC analysis ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S016786550500303X))
* [ ] Adaptive Computation Time for Recurrent Neural Networks ([arXiv](https://arxiv.org/abs/1603.08983))
* [ ] Spatially Adaptive Computation Time for Residual Networks ([arXiv](https://arxiv.org/abs/1612.02297))
* [ ] A mixed-scale dense convolutional neural network for image analysis ([PNAS](http://www.pnas.org/content/115/2/254))
* [x] mixup: Beyond Empirical Risk Minimization ([arXiv](https://arxiv.org/abs/1710.09412))
* Hyper Networks ([blog](http://blog.otoro.net/2016/09/28/hyper-networks/))


## 2018-02
### Study Plan
* [ ] 生物医学成像学导论 ([MIC](topics/mic.md))
    * [ ] Module 2 CT
* [ ] lunglab-keras ([MIC](topics/mic.md))
* [ ] Pointer Network in PyTorch ([AML](topics/advanced_ml.md))

### Reading
* Set / Points
  * [x] Deep Learning with Sets and Point Clouds ([arXiv](https://arxiv.org/abs/1611.04500)) (another paper: Deep Sets)
  * [x] Order Matters: Sequence to sequence for sets ([arXiv](https://arxiv.org/abs/1511.06391))
  * [x] Neural Message Passing for Quantum Chemistry ([arXiv](https://arxiv.org/abs/1704.01212))  
  * [x] PointCNN ([arXiv](https://arxiv.org/abs/1801.07791))
* Interpretability
  * [x] Learning Deep Features for Discriminative Localization (a.k.a CAM) ([arXiv](https://arxiv.org/abs/1512.04150))
  * [x] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization ([arXiv](https://arxiv.org/abs/1610.02391))
  * [x] Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning ([arXiv](https://arxiv.org/abs/1506.02142))
  * A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks ([arXiv](https://arxiv.org/abs/1610.02136))
* [x] Fraternal Dropout ([arXiv](https://arxiv.org/abs/1711.00066))
* [ ] A Tutorial on Variational Bayesian Inference ([pdf](http://www.orchid.ac.uk/eprints/40/1/fox_vbtut.pdf))
* [ ] Tutorial on Variational Autoencoders ([arXiv](https://arxiv.org/abs/1606.05908))

## 2018-01
### MyWeekly
  * [Reinforcement Learning Demystified](weekly/RL_demystified.pdf)
### Study Plan
* [x] ADLxMLDS ([AML](topics/advanced_ml.md))
    * [x] RL (Deep RL + Deep RL2 + Imitation Learning)
    * [x] 5 Attention 
    * [x] 6 Special Networks
    * [x] 7 Tips
    * [x] 10 GAN 
    * [x] 11 GAN for Seq 
    * [x] 12 More GAN 
* [x] CS231n 2017 ([AML](topics/advanced_ml.md))
    * [x] Lecture 12: Visualizing and Understanding ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf))
    * [x] Lecture 16: Adversarial Examples and Adversarial Training ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture16.pdf))
* [x] Atari Game Playing ([AML](topics/advanced_ml.md))
* [ ] Fundamentals of Medical Imaging ([MIC](topics/mic.md))
    * [ ] Chapter 1 
    * [ ] Chapter 2 X-rays
    * [ ] Chapter 3 CT  
* [ ] 生物医学成像学导论 ([MIC](topics/mic.md))
    * [ ] Module 2 CT
* [ ] lunglab-keras ([MIC](topics/mic.md))
### Reading
* Convolutional Invasion and Expansion Networks for Tumor Growth Prediction ([IEEE](http://ieeexplore.ieee.org/document/8110658/)): very simple algorithm, even some bad design... but good at bio-medical explanation. Good feature engineering. 
* Runtime Neural Pruning ([NIPS](https://nips.cc/Conferences/2017/Schedule?showEvent=9006))
* [x] Attention Is All You Need ([arXiv](https://arxiv.org/abs/1706.03762)): 
  - *NB*: very impressive work. Seems to be inspired by Conv Seq2Seq, but more general. 3 key points: 
  - 1) variable-length inputs can be also processed in "attention": softmax(KT.dot(Q)).dot(V) => fix size `d`
  - 2) seq2seq is indeed `encoder output` + `decoder scoring per step`! Can be parallel implemented with masking (seems to come from conv seq2seq?)
  - 3) RNN / Conv architecture can still be used, espeically in decoder
* [x] One Model To Learn Them All ([arXiv](https://arxiv.org/abs/1706.05137)): Xception+Transformer+Sparse MoE in one network. Too big title.
* [x] Convolutional Sequence to Sequence Learning ([arXiv](https://arxiv.org/abs/1705.03122)): Transformer paper seems to absort all of its goodness... 
* [x] Evaluate the Malignancy of Pulmonary Nodules Using the 3D Deep Leaky Noisy-or Network ([arXiv](https://arxiv.org/abs/1711.08324)): DSB2017 1st paper.
* [x] On Bayesian Deep Learning and Deep Bayesian Learning ([YouTube](https://www.youtube.com/watch?v=LVBvJsTr3rg))
* [x] Pointer Networks ([arXiv](https://arxiv.org/abs/1506.03134))
* 动手学深度学习第十三课：正向传播、反向传播和通过时间反向传播 ([YouTube](https://www.youtube.com/watch?v=xPFbbLxegH0&list=PLLbeS1kM6teJqdFzw1ICHfa4a1y0hg8Ax&index=13))
* 动手学深度学习第十六课：词向量（word2vec）([YouTube](https://www.youtube.com/watch?v=C4X0Cb5_FSo&index=16&list=PLLbeS1kM6teJqdFzw1ICHfa4a1y0hg8Ax))