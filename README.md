# Learning Notes by [Jiancheng Yang](https://jiancheng-yang.com/)

Archives: [2018](archive/note18.md) | [before 2018](archive/before18.md)

<!-- ## Study Plan -->
<!-- AML - [Advanced Machine Learning](topics/advanced_ml.md) | MIC - [Medical Image Computing](topics/mic.md) | Prog - [Programming](topics/programming.md) -->

Related projects: 
* [`M3DV/ai-deadlines`](https://github.com/M3DV/ai-deadlines): Top AI deadline countdowns (with an emphasis on computer vision and medical images).
* [`M3DV/Kickstart`](https://github.com/M3DV/Kickstart): Study route for learners in machine learning / deep learning / computer vision.
<!-- * [`M3DV/Readiness-Seminar`](https://github.com/M3DV/Readiness-Seminar): A cooperative paper list on 3D vision and deep learning robustness. -->

## 2021-01
### Research 
* [x] End-to-End Learning on 3D Protein Structure for Interface Prediction ([NeurIPS'19](https://papers.nips.cc/paper/2019/hash/6c7de1f27f7de61a6daddfffbe05c058-Abstract.html))
* [x] ATOM3D: Tasks On Molecules in Three Dimensions ([arXiv](https://arxiv.org/abs/2012.04035))
* [ ] Deep Learning of High-Order Interactions for Protein Interface Prediction ([KDD'20](https://arxiv.org/abs/2007.09334))
* [ ] LambdaNetworks: Modeling long-range Interactions without Attention ([ICLR'21 (submission likely to be accepted)](https://openreview.net/forum?id=xTJEN-ggl1b))
* [ ] Neural Controlled Differential Equations for Irregular Time Series ([NeurIPS'20](https://arxiv.org/pdf/2005.08926v1.pdf))
* [ ] Countdown Regression: Sharp and Calibrated Survival Predictions ([UAI'19](https://arxiv.org/abs/1806.08324))
* [ ] Latent ODEs for Irregularly-Sampled Time Series ([NeurIPS'19](https://arxiv.org/abs/1907.03907))
* [ ] Alleviating Class-wise Gradient Imbalance for Pulmonary Airway Segmentation ([arXiv]())
* [ ] Learning Tubule-Sensitive CNNs for Pulmonary Airway and Artery-Vein Segmentation in CT ([arXiv](https://arxiv.org/pdf/2012.05767.pdf))
* [ ] Segmenting and tracking cell instances with cosine embeddings and recurrent hourglass networks ([MedIA'19](https://doi.org/10.1016/j.media.2019.06.01))
* [ ] What is being transferred in transfer learning? ([NeurIPS'20](https://proceedings.neurips.cc/paper/2020/file/0607f4c705595b911a4f3e7a127b44e0-Paper.pdf))
* [ ] Transfusion: Understanding Transfer Learning for Medical Imaging ([NeurIPS'19](https://papers.nips.cc/paper/8596-transfusion-understanding-transfer-learning-for-medical-imaging))
* [ ] EllipTrack: A Global-Local Cell-Tracking Pipeline for 2D Fluorescence Time-Lapse Microscopy ([Cell Reports'20](https://www.sciencedirect.com/science/article/pii/S2211124720309694))
* [ ] DeepCenterline: a Multi-task Fully Convolutional Network for Centerline Extraction ([IPMI'19](https://arxiv.org/abs/1903.10481))
* [ ] Deep Distance Transform for Tubular Structure Segmentation in CT Scans ([CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Deep_Distance_Transform_for_Tubular_Structure_Segmentation_in_CT_Scans_CVPR_2020_paper.html))
* [ ] Tracing in 2D to reduce the annotation effort for 3D deep delineation of linear structures ([MedIA'20](https://www.sciencedirect.com/science/article/pii/S1361841519301306))
* [ ] Rapid vessel segmentation and reconstruction of head and neck angiograms using 3D convolutional neural network ([Nature Communications'20](https://www.nature.com/articles/s41467-020-18606-2))

### Study
* [ ] [Differential Equations for Engineers](https://www.coursera.org/learn/differential-equations-engineers/home/welcome)
  * [ ] W1: First-Order Differential Equations
  * [ ] W2: Homogeneous Linear Differential Equations
  * [ ] W3: Inhomogeneous Linear Differential Equations
  * [ ] W4: The Laplace Transform and Series Solution Methods
  * [ ] W5: Systems of Differential Equations
  * [ ] W6: Partial Differential Equations

## 2020-12
I recently read a few about the Auto Augmentation techniques, there is a [summary](notes/autoaug.md).

### Research
* [x] SlowFast Networks for Video Recognition ([ICCV'19 Oral](https://arxiv.org/abs/1812.03982))
* [x] ~Temporal Interlacing Network ([AAAI'20](https://arxiv.org/abs/2001.06499)): extension of TSM, something similar to RubiksNet (input-conditional shift). Bad writing.
* [x] ~Temporal Pyramid Network for Action Recognition ([CVPR'20](https://arxiv.org/abs/2004.03548))
* [x] RandAugment: Practical automated data augmentation with a reduced search space ([NeurIPS'20](https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf)): a surprisingly simple method which beats several learnable AutoAug. 
* [x] ~Adversarial AutoAugment ([ICLR'20](https://arxiv.org/abs/1912.11188))
* [x] Fast AutoAugment ([NeurIPS'19](https://papers.nips.cc/paper/2019/hash/6add07cf50424b14fdf649da87843d01-Abstract.html))
* [x] Faster AutoAugment: Learning Augmentation Strategies Using Backpropagation ([ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700001.pdf)) ([code](https://github.com/moskomule/dda))
* [x] DADA: Differentiable Automatic Data Augmentation ([ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670579.pdf))
* [x] Differentiable Augmentation for Data-Efficient GAN Training ([NeurIPS'20](https://papers.nips.cc/paper/2020/hash/55479c55ebd1efd3ff125f1337100388-Abstract.html)): how to apply augmentation in GAN, both D and G. Back-prop T(G) need T to be diff.
* [x] Improving Auto-Augment via Augmentation-Wise Weight Sharing ([NeurIPS'20](https://papers.nips.cc/paper/2020/hash/dc49dfebb0b00fd44aeff5c60cc1f825-Abstract.html)): looks like a Fast AA scheme.
* [x] Test-Time Training with Self-Supervision for Generalization under Distribution Shifts ([ICML'20](https://arxiv.org/abs/1909.13231)): surprisingly simple method, do auxillary self-supervised task of rotation prediction with a branch, achieving good performance. Writing is hard to follow.
* [x] ~Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
* [x] ~Benchmarking Adversarial Robustness on Image Classification ([CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Benchmarking_Adversarial_Robustness_on_Image_Classification_CVPR_2020_paper.pdf))
* [x] Realistic Adversarial Data Augmentation for MR Image Segmentation ([MICCAI'20](https://arxiv.org/abs/2006.13322)): multiplicative adversarial attack with a spatial priors to assist training, like VAT.
* [x] When and Why Test-Time Augmentation Works ([arXiv](https://arxiv.org/abs/2006.13322)): Analysis of TTA + a simple aggregation of TTA
* [x] Learning 3D Features with 2D CNNs via Surface Projection for CT Volume Segmentation ([MICCAI'20](https://link.springer.com/chapter/10.1007%2F978-3-030-59719-1_18)): learning deformation from UV maps, it is not real 3D features.
* [x] HFA-Net: 3D Cardiovascular Image Segmentation with Asymmetrical Pooling and Content-Aware Fusion ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_84))
* [x] 3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training ([WACV'20](https://arxiv.org/abs/1811.12506))
* [x] KiU-Net: Towards Accurate Segmentation of Biomedical Images using Over-complete Representations ([MICCAI'20](https://arxiv.org/abs/2006.04878)): something like FishNet and Hourglass.
* [x] Graph Cross Networks with Vertex Infomax Pooling ([NeurIPS'20](https://arxiv.org/abs/2010.01804))
* [x] MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation from EM Images ([MICCAI'20](https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf))
* [x] AbdomenCT-1K: Is Abdominal Organ Segmentation A Solved Problem? ([arXiv](https://arxiv.org/abs/2010.14808))
* [x] ～Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution ([ICCV'19](https://arxiv.org/abs/1904.05049))
* [x] ~Decoupling Representation and Classifier for Long-Tailed Recognition ([ICLR'20](https://arxiv.org/abs/1910.09217))
* [x] ~Stacked U-Nets: A No-Frills Approach to Natural Image Segmentation ([arXiv](https://arxiv.org/abs/1804.10343))
* [x] ~GhostNet: More Features from Cheap Operations ([CVPR'20](https://arxiv.org/abs/1911.11907))
* [x] Recalibrating 3D ConvNets with Project & Excite ([TMI'20](https://arxiv.org/abs/2002.10994)): a not so naive extension of SE block for 3D conv.
* [x] ～ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks ([ICCV'19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf))
* [x] ~CT-ORG, a new dataset for multiple organ segmentation in computed tomography ([Scientific Data'20](https://www.nature.com/articles/s41597-020-00715-8#Sec9))
* [x] Self-supervised Feature Learning for 3D Medical Images by Playing a Rubik's Cube ([MICCAI'19](https://arxiv.org/abs/1910.02241))
* [x] Rubik’s Cube+: A self-supervised feature learning framework for 3D medical image analysis ([MedIA'20](https://doi.org/10.1016/j.media.2020.101746))
* [x] ~Efficient Multiple Organ Localization in CT Image Using 3D Region Proposal Network ([TMI'19](https://ieeexplore.ieee.org/document/8625393)): abdomen(-chest) CT organ localization annotations for LiTS dataset with a baseline 3D detection.
* [x] ~Deep Image Prior ([CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf)): overfitting a single image jsut like SinGAN for imaging application.
* [x] Self-supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation ([ECCV'20](https://arxiv.org/pdf/2007.09886.pdf))
* [x] Data augmentation using learned transforms for one-shot medical image segmentation ([CVPR'19](https://arxiv.org/abs/1902.09383))
* [x] ~Reformer: The Efficient Transformer ([ICLR'20](https://openreview.net/forum?id=rkgNKkHtvB)): LSH-based (sparse) top-K softmax for attention + RevNet-based transformer. Writing is hard to follow.
* [x] Rethinking Attention with Performers ([ICLR'21 (submission likely to be accepted)](https://openreview.net/forum?id=Ua6zuk0WRH)): the super power of mateix decomposition and stochastic appromixation. Lots of math makes it hard to follow (the paper structure is not friendly to ones without solid theoretical reading, as you could find much important information in unexcepted place, e.g., important model variant Performer-ReLU first appears in experiment section, SMREG regularization technique in theorectical analysis), but it is indeed a good paper, and empirically effective in the paper experiments. At least a Spotlight, I guess it to be **a strong candidate for ICLR'21 best paper**. It can be: 1) fast and efficient approximation of standard attention (O(N2)->O(N)): linear complexity; 2) converting an existing Transformer into Performers and **funetuning** (it is necessary) makes a fast attention model; 3) generalized attention with ReLU kernel seems always better than Softmax (it is also easier to implement)
* [ ] LambdaNetworks: Modeling long-range Interactions without Attention ([ICLR'21 (submission likely to be accepted)](https://openreview.net/forum?id=xTJEN-ggl1b))
* [ ] End-to-End Learning on 3D Protein Structure for Interface Prediction ([NeurIPS'19](https://papers.nips.cc/paper/2019/hash/6c7de1f27f7de61a6daddfffbe05c058-Abstract.html))
* [ ] ATOM3D: Tasks On Molecules in Three Dimensions ([arXiv](https://arxiv.org/abs/2012.04035))
* [ ] Deep Learning of High-Order Interactions for Protein Interface Prediction ([KDD'20](https://arxiv.org/abs/2007.09334))


### Study
* [ ] [GAMES101-现代计算机图形学入门](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html): It looks very good as a Chinese course for CG. I decide to finish this course.
  <!-- * [x] Lecture 01 Overview of Computer Graphics -->
  <!-- * [x] ~Lecture 02 Review of Linear Algebra -->
  * [x] Lecture 03 Transformation
  * [ ] Lecture 04 Transformation Cont.
  * [ ] Lecture 05 Rasterization 1 (Triangles)
  * [ ] Lecture 06 Rasterization 2 (Antialiasing and Z-Buffering)
* [ ] [Differential Equations for Engineers](https://www.coursera.org/learn/differential-equations-engineers/home/welcome)

## 2020-11
### Research
* [x] An objective comparison of cell-tracking algorithms ([Nature Methods](https://www.nature.com/articles/nmeth.4473))
* [ ] Segmenting and tracking cell instances with cosine embeddings and recurrent hourglass networks ([Medical Image Analysis](https://doi.org/10.1016/j.media.2019.06.01))
* [x] Born Again Neural Networks ([ICML'18](http://proceedings.mlr.press/v80/furlanello18a.html))
* [x] Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation ([ICCV'19](https://doi.org/10.1109/ICCV.2019.00381)): deep supervision + (self-)distillation
* [x] Task-Oriented Feature Distillation ([NeurIPS'20](https://papers.nips.cc/paper/2020/file/a96b65a721e561e1e3de768ac819ffbb-Paper.pdf))
* [x] ~NBDT: Neural-Backed Decision Trees ([arXiv](https://arxiv.org/abs/2004.00221))
* [x] Square Attack: a query-efficient black-box adversarial attack via random search ([ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680477.pdf)): generate heavily hand-crafted perturbation sensitive to CNNs (somehow "adversarially structured noises"). Simple idea but lots of trials. Paper is not easy to read, maybe step-by-step illustration of the perturbation generation is better. Very impressive results without surrogate models, though our NeurIPS'20 paper works better still ;) (we use surrogate models).
* [x] Learning Loss for Test-Time Augmentation ([NeurIPS'20](https://arxiv.org/abs/2010.11422)): Imitation learning for test-time augmentation with a ranking loss. Writing is not good.
* [x] ~RubiksNet: Learnable 3D-Shift for Efficient Video Action Recognition ([ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640494.pdf)): Shift in 3D, using interpolation (like our AlignShift) and temperture annealing.
* [x] MAST: A Memory-Augmented Self-supervised Tracker ([CVPR'20](https://arxiv.org/abs/2002.07793)): interesting paper with good performance, sadly there seems some details missing in the paper for full understanding.

### Study
* [ ] [GAMES101-现代计算机图形学入门](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html): It looks very good as a Chinese course for CG. I decide to finish this course.
  <!-- * [x] Lecture 01 Overview of Computer Graphics -->
  <!-- * [x] ~Lecture 02 Review of Linear Algebra -->
  * [ ] Lecture 03 Transformation
  * [ ] Lecture 04 Transformation Cont.
  * [ ] Lecture 05 Rasterization 1 (Triangles)
  * [ ] Lecture 06 Rasterization 2 (Antialiasing and Z-Buffering)

## 2020-10
### Research
* [x] Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion ([CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chibane_Implicit_Functions_in_Feature_Space_for_3D_Shape_Reconstruction_and_CVPR_2020_paper.pdf)): A  very good paper with journal-level reviews of related work. Good method modified from DISN, easy to implement, and also good results.
* [x] DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction ([NeurIPS'19](http://papers.nips.cc/paper/8340-disn-deep-implicit-surface-network-for-high-quality-single-view-3d-reconstruction)): interesting paper for single image recontruction, the base of IF-Net. Some writing parts are hard to follow. 
* [x] Point-Voxel CNN for Efficient 3D Deep Learning ([NeurIPS'19](https://arxiv.org/abs/1907.03739)): solid work for point cloud understanding.
* [ ] MAST: A Memory-Augmented Self-supervised Tracker ([CVPR'20](https://arxiv.org/abs/2002.07793>)) ([code](https://github.com/zlai0/MAST))
* [ ] 3D Shape Reconstruction from Vision and Touch ([NeurIPS'20](https://arxiv.org/abs/2007.03778))
Self-Supervising Point Cloud Representation Learning by Shape-Correction
* [ ] ShapeFlow: Learnable Deformations Among 3D Shapes ([NeurIPS'20](https://arxiv.org/abs/2006.07982))
* [ ] What shapes feature representations? Exploring datasets, architectures, and training ([NeurIPS'20](https://arxiv.org/abs/2006.12433))
* [ ] MeshSDF: Differentiable Iso-Surface Extraction ([NeurIPS'20](https://arxiv.org/abs/2006.03997))
* [ ] UCLID-Net: Single View Reconstruction in Object Space
 ([NeurIPS'20](https://arxiv.org/abs/2006.03817))
* [ ] Continuous Surface Embeddings ([NeurIPS'20](?))
* [ ] Neural Unsigned Distance Fields for Implicit Function Learning ([NeurIPS'20](?))
* [ ] Graph Cross Networks with Vertex Infomax Pooling ([NeurIPS'20](https://arxiv.org/abs/2010.01804))

### Study
* [ ] [GAMES101-现代计算机图形学入门](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html): It looks very good as a Chinese course for CG. I decide to finish this course.
  * [x] Lecture 01 Overview of Computer Graphics
  * [x] ~Lecture 02 Review of Linear Algebra
  * [ ] Lecture 03 Transformation
  * [ ] Lecture 04 Transformation Cont.
  * [ ] Lecture 05 Rasterization 1 (Triangles)
  * [ ] Lecture 06 Rasterization 2 (Antialiasing and Z-Buffering)
* [ ] [AI for Medical Prognosis](https://www.coursera.org/learn/ai-for-medical-prognosis/home/welcome): quick review and practice.
  * [ ] Week 1 Linear prognostic models
  * [ ] Week 2 Prognosis with Tree-based models
  <!-- * [ ] Week 3 Survival Models and Time -->
  <!-- * [ ] Week 4 Build a risk model using linear and tree-based models -->

## 2020-08
There is time with no updating the repo... I will soon re-launch it. 

## 2020-07

### Reading
* [ ] Decoupling Representation and Classifier for Long-Tailed Recognition ([ICLR'20](https://arxiv.org/abs/1910.09217))
* [ ] Reformer: The Efficient Transformer ([ICLR'20](https://openreview.net/forum?id=rkgNKkHtvB))
* [ ] ~What Makes for Good Views for Contrastive Learning? ([arXiv](https://arxiv.org/abs/2005.10243)) ([code](https://github.com/HobbitLong/PyContrast))
* [x] ~Prototypical Contrastive Learning of Unsupervised Representations ([arXiv](https://arxiv.org/abs/2005.04966)): MoCo + clustering-based "supervised" learning (there seems paper on this, but I can not recall).
* [x] ~Momentum Contrast for Unsupervised Visual Representation Learning ([arXiv](https://arxiv.org/abs/1911.05722/)) ([PyTorch code](https://github.com/facebookresearch/moco))
* [x] ~Improved Baselines with Momentum Contrastive Learning (a.k.a MOCOv2) ([arXiv](https://arxiv.org/abs/2003.04297))
* [x] ~A Simple Framework for Contrastive Learning of Visual Representations ([arXiv](https://arxiv.org/abs/2002.05709)) ([code](https://github.com/google-research/simclr))
* [x] Predicting tumour mutational burden from histopathological images using multiscale deep learning ([Nat MI](https://www.nature.com/articles/s42256-020-0190-5)) ([biorxiv]https://www.biorxiv.org/content/10.1101/2020.06.15.153379v1))


## 2020-06
There is time with no updating the repo... I will restart it.

I got one more (co-first authored) paper accepted by MICCAI'20, which results in 4 totally. How lucky!

### Reading
* [x] Using transfer learning on whole slide images to predict tumor mutational burden in bladder cancer patients ([bioRxiv](https://www.biorxiv.org/content/10.1101/554527v1))
* [x] Guided evolutionary strategies: Augmenting random search with surrogate gradients ([ICML'19](http://proceedings.mlr.press/v97/maheswaranathan19a/maheswaranathan19a.pdf))
* [x] UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation ([ICASSP'20](http://arxiv.org/abs/2004.08790))
* [x] Light-Weight Hybrid Convolutional Network for Liver Tumor Segmentation ([IJCAI'19](https://www.ijcai.org/Proceedings/2019/0593.pdf))
* [ ] Decoupling Representation and Classifier for Long-Tailed Recognition ([ICLR'20](https://arxiv.org/abs/1910.09217))
* [x] Deep Learning Based Rib Centerline Extraction and Labeling ([MICCAI'18 MSKI 2018 Workshop](https://arxiv.org/abs/1809.07082)): it is in fact a very interesting paper, although towards small audience. Solid methods (although not elegant with lots of hand-crafted "engineering"), good results. It deserves a better publication venue and more citation if presented with good story. The authors seem to assume the audience to have much background in chest CT image analysis.
* [x] X3D: Expanding Architectures for Efficient Video Recognition ([CVPR'20 oral](https://arxiv.org/abs/2004.04730)): a good paper with solid results. Straightforward method: simple and compact basis network to expand in terms of temporal, spatial, network width and depth, etc., just like EfficientNet for videos. However, the writing is somehow misleading in certain words (although it is better to understand with the supplementary materials). There is also typos (I read the arXiv version).  
* [ ] Reformer: The Efficient Transformer ([ICLR'20](https://openreview.net/forum?id=rkgNKkHtvB))
* [x] Embracing Imperfect Datasets: A Review of Deep Learning Solutions for Medical Image Segmentation ([MedIA](https://arxiv.org/abs/1908.10454)): a systematic review for scarce and weak annotations. Good to relate the methods, but seems not surprising for me. PS: it is very long (30+ pages). 
* [ ] ~What Makes for Good Views for Contrastive Learning? ([arXiv](https://arxiv.org/abs/2005.10243)) ([code](https://github.com/HobbitLong/PyContrast))
* [ ] ~Prototypical Contrastive Learning of Unsupervised Representations ([arXiv](https://arxiv.org/abs/2005.04966))
* [ ] ~Momentum Contrast for Unsupervised Visual Representation Learning ([arXiv](https://arxiv.org/abs/1911.05722/)) ([PyTorch code](https://github.com/facebookresearch/moco))
* [ ] ~Improved Baselines with Momentum Contrastive Learning (a.k.a MOCOv2) ([arXiv](https://arxiv.org/abs/2003.04297))
* [ ] ~A Simple Framework for Contrastive Learning of Visual Representations ([arXiv](https://arxiv.org/abs/2002.05709)) ([code](https://github.com/google-research/simclr))


## 2020-05
Three papers (first author) were accepted by MICCAI'20! All are early accepted! Lucky ;)

## 2020-04

One paper on noisy labels for detection (our 1st runner up solution of MICCAI DigestPath2019 challenge) was accepted by Neurpcomputing.

* Medical 
  * [ ] Biologically-Constrained Graphs for Global Connectomics Reconstruction ([CVPR'19](https://donglaiw.github.io/paper/2019_cvpr_skel.pdf))
  * [ ] Data augmentation using learned transformations for one-shot medical image segmentation ([CVPR'19](https://arxiv.org/abs/1902.09383))
  * [ ] Clinically applicable deep learning framework for organs at risk delineation in CT images ([Nature Machine Intelligence](https://www.nature.com/articles/s42256-019-0099-z))
  * [ ] ~Transfusion: Understanding Transfer Learning for Medical Imaging ([NeurIPS'19](https://arxiv.org/abs/1902.07208))
* Adversarial Examples
  * [ ] Adversarial Examples Are Not Bugs, They Are Features ([arXiv](https://arxiv.org/abs/1905.02175))
  * [ ] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
  * [ ] Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors ([ICLR'19](https://openreview.net/forum?id=BkMiWhR5K7))
  * [ ] Improving Black-box Adversarial Attacks with a Transfer-based Prior ([NeurIPS'19](https://arxiv.org/abs/1906.06919)) ([code](https://github.com/prior-guided-rgf/Prior-Guided-RGF))
* Misc
  * [x] Guided evolutionary strategies: Augmenting random search with surrogate gradients ([ICML'19](http://proceedings.mlr.press/v97/maheswaranathan19a/maheswaranathan19a.pdf))
  * [ ] Momentum Contrast for Unsupervised Visual Representation Learning ([arXiv](https://arxiv.org/abs/1911.05722/))
  * [ ] GhostNet: More Features from Cheap Operations ([CVPR'20](httpse://arxiv.org/abs/1911.11907))
  * [ ] Reformer: The Efficient Transformer ([ICLR'20](https://openreview.net/forum?id=rkgNKkHtvB)) ([trax code](https://github.com/google/trax/tree/master/trax/models/reformer)) ([PyTorch code](https://github.com/lucidrains/reformer-pytorch))



## 2020-03
Preparing ECCV'20 and MICCAI'20 submissions, plus a clinical journal submission.

### Reading
* Medical 
  * [x] Holistic and Comprehensive Annotation of Clinically Significant Findings on Diverse CT Images: Learning from Radiology Reports and Label Ontology ([CVPR'19](https://arxiv.org/abs/1904.04661))
  * [x] Deep Lesion Graphs in the Wild: Relationship Learning and Organization of Significant Radiology Image Findings in a Diverse Large-scale Lesion Database ([CVPR'18](https://arxiv.org/abs/1711.10535))
  * [x] Deep Probabilistic Modeling of Glioma Growth ([MICCAI'19](https://arxiv.org/abs/1907.04064))
  * [x] MULAN: Multitask Universal Lesion Analysis Network for Joint Lesion Detection, Tagging, and Segmentation ([MICCAI'19](https://arxiv.org/abs/1908.04373))
  * [x] 3D Context Enhanced Region-based Convolutional Neural Network for End-to-End Lesion Detection ([MICCAI'18](https://arxiv.org/abs/1806.09648))
  * [x] MVP-Net: Multi-view FPN with Position-aware Attention for Deep Universal Lesion Detection ([MICCAI'19](https://arxiv.org/abs/1909.04247/))
* Misc
  * [x] Neural Ordinary Differential Equations ([NeurIPS'18 best](https://arxiv.org/abs/1806.07366)): it is absolutely a very innotative paper. I did not care the proof on how to get the gradient of ODE. There seems lots of applications.
  * [ ] Momentum Contrast for Unsupervised Visual Representation Learning ([arXiv](https://arxiv.org/abs/1911.05722/))
  * [ ] GhostNet: More Features from Cheap Operations ([CVPR'20](httpse://arxiv.org/abs/1911.11907))
  * [ ] Reformer: The Efficient Transformer ([ICLR'20](https://openreview.net/forum?id=rkgNKkHtvB)) ([trax code](https://github.com/google/trax/tree/master/trax/models/reformer)) ([PyTorch code](https://github.com/lucidrains/reformer-pytorch))


## 2020-02

China is in the shadow of 2019-nCoV. I'm super busy but upset. Hope everything works well and I could be better me.

We have submitted a paper to ICML'20. 

One paper on 3D pose estimation was accepted by CVPR'20. 

Besides, we are informed to be (very) likely to host a MICCAI challenge this year. I am the leading organizer.

### Reading
  * [x] Rectified Cross-Entropy and Upper Transition Loss for Weakly Supervised Whole Slide Image Classifier ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_39))
  * [x] PointRend: Image Segmentation as Rendering ([arXiv](https://arxiv.org/abs/1912.08193))
  * [x] Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation ([ICCV'19](https://arxiv.org/abs/1811.07456)) ([code](https://github.com/jihanyang/AFN))
  * [x] ~Attract or Distract: Exploit the Margin of Open Set ([ICCV'19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_Attract_or_Distract_Exploit_the_Margin_of_Open_Set_ICCV_2019_paper.pdf))
  * [x] Class-Balanced Loss Based on Effective Number of Samples ([CVPR'19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf))
  * [x] Large-Scale Long-Tailed Recognition in an Open World ([CVPR'19](https://arxiv.org/abs/1904.05160))
  * [x] ~CARAFE: Content-Aware ReAssembly of FEatures ([ICCV'19](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html)): it resemables Meta-SR.


## 2020-01

Super busy but less productive. Chinese new year is comming, I need focus and hard work! I wish all Chinese could be safe under the threat of 2019-nCoV, and the world could be fine.

One paper was accepted by ISBI'20.

### Study
* [x] 心理学概论 ([Xuetangx](https://next.xuetangx.com/learn/THU07111000416/THU07111000416/1516445/video/1395645)): Module 1,2,3

## 2019-12
See u at RSNA 2019, Chicago! 

This month I worked on staffs, without complete paper reading... Nevertheless, I have completed the CVPR 2020 reviewing.



## 2019-11
Prepare CVPR submissions...

### Reading
* [x] SinGAN: Learning a Generative Model from a Single Natural Image ([ICCV'19 best paper](http://webee.technion.ac.il/people/tomermic/SinGAN/SinGAN.htm)): good method, very strong results and application (especially the application on single image annimation, very impressive).
* [x] CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features ([ICCV'19](https://arxiv.org/abs/1905.04899)): simple method, good results. 

## 2019-10
I have just submitted papers to ISBI 2019 and will be attending MICCAI 2019 in Shenzhen, China.

### Reading

* Medical (Emphasis on MICCAI'19)
  * [x] Overfitting of neural nets under class imbalance: Analysis and improvements for segmentation ([MICCAI'19](https://arxiv.org/abs/1907.10982)): very insightful. Less data contributes to low sensitivity / recall to foreground rather than precision. 
  * [x] DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation ([MICCAI'15](https://link.springer.com/chapter/10.1007%2F978-3-319-24553-9_68)) ([dataset](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT): a.k.a. NIH Pancreas CT)
  * [x] Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation ([CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.pdf)) ([code](https://github.com/twni2016/OrganSegRSTN_PyTorch))
  * [ ] Harnessing 2D Networks and 3D Features for Automated Pancreas Segmentation from Volumetric CT Images ([MICCAI'19](https://link.springer.com/chapter/10.1007%2F978-3-030-32226-7_38))
  * [ ] Fully Automated Pancreas Segmentation with Two-Stage 3D Convolutional Neural Networks ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_23))
  * [ ] Searching Learning Strategy with Reinforcement Learning for 3D Medical Image Segmentation ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_1))
  * [ ] One Network to Segment Them All: A General, Lightweight System for Accurate 3D Medical Image Segmentation ([MICCAI'19](https://link.springer.com/chapter/10.1007%2F978-3-030-32245-8_4)) ([code](https://github.com/perslev/MultiPlanarUNet))
  * [ ] 3D Tiled Convolution for Effective Segmentation of Volumetric Medical Images ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_17)) ([code](https://github.com/guoyanzheng/LPNet))
  * [ ] 3D U2-Net: A 3D Universal U-Net for Multi-Domain Medical Image Segmentation ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_33)) ([code](https://github.com/huangmozhilv/u2net_torch/))
  * [ ] Scalable Neural Architecture Search for 3D Medical Image Segmentation ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_25))
  * [ ] Self-supervised Feature Learning for 3D Medical Images by Playing a Rubik’s Cube ([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_46))
  * [ ] Resource Optimized Neural Architecture Search for 3D Medical Image Segmentation([MICCAI'19](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_26))
  * [x] NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation ([MICCAI'19](https://arxiv.org/abs/1907.11320)) ([code](https://github.com/uci-cbcl/NoduleNet))
  * [ ] Clinically applicable deep learning framework for organs at risk delineation in CT images ([Nature Machine Intelligence](https://www.nature.com/articles/s42256-019-0099-z))
  * [ ] ~Transfusion: Understanding Transfer Learning for Medical Imaging ([NeurIPS'19](https://arxiv.org/abs/1902.07208))
  * [x] NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation ([MICCAI'19](https://arxiv.org/abs/1907.11320)) ([code](https://github.com/uci-cbcl/NoduleNet))
* Graph
  * [x] ~Deep Generative Models for Graphs: Methods & Applications ([ICLR'19 tutorial](https://slideslive.com/38915801/deep-graph-generative-models-methods-applications))
  * [x] ~Representation Learning on Networks ([WWW'18 tutorial](http://snap.stanford.edu/proj/embeddings-www/))
  * [x] ~网络表示学习综述 ([pdf](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/sc2017_nrl.pdf))

## 2019-09

### Reading

* Medical
  * [ ] Overfitting of neural nets under class imbalance: Analysis and improvements for segmentation ([MICCAI'19](https://arxiv.org/abs/1907.10982))
  * [x] Elastic Boundary Projection for 3D Medical Imaging Segmentation ([CVPR'19](https://arxiv.org/abs/1812.00518)): it is still 2D segmentation, while iteratively. Only eligible for segmentation task. The inference speed should be very low. Now aplicable for irregular geometric shape. The whole idea is very related to shape modeling / human-in-the-loop segmentation.
  * [x] Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis ([MICCAI'19](https://paperswithcode.com/paper/models-genesis-generic-autodidactic-models)): self-supervised learning on MIC.
  * [x] Med3D: Transfer Learning for 3D Medical Image Analysis ([arXiv](https://arxiv.org/abs/1904.00625)) ([code](https://github.com/Tencent/MedicalNet)): easy but useful. Better carefully using.
  * [x] 3D Anisotropic Hybrid Network: Transferring Convolutional Features from 2D Images to 3D Anisotropic Volumes ([MICCAI'18](https://arxiv.org/abs/1711.08580)): important direction but not cool: not elegant solution, bad writing. 
  * [x] H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes ([TMI'18](https://arxiv.org/abs/1709.07330)) ([code](https://github.com/xmengli999/H-DenseUNet)): good result as a strong baseline but not elegent method.
  * [x] 3DFPN-HS2: 3D Feature Pyramid Network Based High Sensitivity and Specificity Pulmonary Nodule Detection ([MICCAI'19](https://arxiv.org/abs/1906.03467)): very high performance.
  * [ ] ~Transfusion: Understanding Transfer Learning for Medical Imaging ([NeurIPS'19](https://arxiv.org/abs/1902.07208))
  * [ ] NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation ([MICCAI'19](https://arxiv.org/abs/1907.11320)) ([code](https://github.com/uci-cbcl/NoduleNet))
  * [x] Thickened 2D Networks for 3D Medical Image Segmentation ([arXiv](https://arxiv.org/abs/1904.01150))
  * [x] Bridging the Gap Between 2D and 3D Organ Segmentation with Volumetric Fusion Net ([MICCAI'18](https://arxiv.org/abs/1804.00392))
  * [x] V-NAS: Neural Architecture Search for Volumetric Medical Image Segmentation ([arXiv](https://arxiv.org/pdf/1906.02817.pdf)): search 2D / 3D CNN kernels.
  * [x] A New Ensemble Learning Framework for 3D Biomedical Image Segmentation ([AAAI'19](https://arxiv.org/abs/1812.03945)): VFN + multi-view consistency training loss.
* Graph
  * [ ] Deep Generative Models for Graphs: Methods & Applications ([ICLR'19 tutorial](https://slideslive.com/38915801/deep-graph-generative-models-methods-applications))
  * [ ] Representation Learning on Networks ([WWW'18 tutorial](http://snap.stanford.edu/proj/embeddings-www/))
  * [ ] ~网络表示学习综述 ([pdf](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/sc2017_nrl.pdf))
* Adversarial Examples
  * [ ] Adversarial Examples Are Not Bugs, They Are Features ([arXiv](https://arxiv.org/abs/1905.02175))
  * [ ] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
* Vision
  * [x] Expectation-Maximization Attention Networks for Semantic Segmentation ([ICCV'19](https://arxiv.org/abs/1907.13426)) ([code](https://github.com/XiaLiPKU/EMANet)): I have thought about the "EMA" mechanism (I even wrote the very similar code, not on segmentation task). However, the authors make the idea work, by making the initial base vectors independent on data, while dependent on dataset. I think it is the key.
* Reinforcement Learning
  * [ ] Recurrent World Models Facilitate Policy Evolution ([NIPS'18](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf)) ([website](https://worldmodels.github.io/))
  * [ ] ~Continuous control with deep reinforcement learning (a.k.a. DDPG) ([ICLR'16](https://arxiv.org/abs/1509.02971))
* Misc
  * [x] Autoregressive Convolutional Neural Networks for Asynchronous Time Series ([ICML'18](http://proceedings.mlr.press/v80/binkowski18a/binkowski18a.pdf))
  * [x] Hidden Technical Debt in Machine Learning Systems ([NIPS'15](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)): worth reading but difficult (for me), because of lots of software engineering terms. 


## 2019-08
Work hard for T-PAMI and CVPR!!!!


### Reading
* Medical
  * [x] End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography ([Nat Medicine](https://www.nature.com/articles/s41591-019-0447-x)): train existing methods on NLST, on healthy (95%+)patients without intermediate results (complete detection results). It is not real world. It is not practical indeed (only produce final diagnosis, with very limited intermediate outputs).
  * [x] Signet Ring Cell Detection With a Semi-supervised Learning Framework ([IPMI'19](https://arxiv.org/abs/1907.03954))
  * [x] Attention U-Net: Learning Where to Look for the  ([MIDL'18](https://openreview.net/pdf?id=Skft7cijM))
  * [x] UNet++: A Nested U-Net Architecture for Medical Image Segmentation ([MICCAI'18 Workshop](https://arxiv.org/abs/1807.10165)): something similar to DLA.
  * [ ] Overfitting of neural nets under class imbalance: Analysis and improvements for segmentation ([MICCAI'19](https://arxiv.org/abs/1907.10982))
* Uncertainty
  * [x] PHiSeg: Capturing Uncertainty in Medical Image Segmentation ([MICCAI'19](https://arxiv.org/abs/1906.04045)) ([code](https://github.com/baumgach/PHiSeg-code)): good results, enhanced Prob-UNet. Multi-scale prior encoding for ambiguous segmentation.
  * [x] Direct Uncertainty Prediction for Medical Second Opinions ([ICML'19](https://arxiv.org/abs/1807.01771)) ([Supp](http://proceedings.mlr.press/v97/raghu19a/raghu19a-supp.pdf)): it is not an easy-to-follow paper, though the core idea is very simple... it needs ground truth disagreement. It does not well leverage the classification and uncertainty prediction task (joint training drops the performance). Anyway, it is not easy to follow, thus I did not read all the experiment part.
  * [ ] ~ Who Said What: Modeling Individual Labelers Improves Classification ([ICLR'17 Workshop->AAAI'18](https://arxiv.org/abs/1703.08774)): Google JAMA paper dataset. Just like my multi-softmax idea. With a little bit improvement in performance in this dataset. EM on the labels is also referred.
  * [ ] ~Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples ([ICLR'18](https://arxiv.org/abs/1711.09325))
  * [ ] ~A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks ([NeurIPS'18](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks))
  * [ ] ~Hierarchical Novelty Detection for Visual Object Recognition ([CVPR'18](https://arxiv.org/abs/1804.00722))
* Vision
  * [x] Meta-SR: A Magnification-Arbitrary Network for Super-Resolution ([CVPR'19](https://arxiv.org/abs/1903.00875)): very bad writing...
  * [x] Gradient Harmonized Single-stage Detector ([AAAI'19](https://arxiv.org/abs/1811.05181)): it seems a good and effective paper. Surprisingly effective on my dataset.
  * [x] Deep Layer Aggregation ([CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf))
  * [x] SSD: Single Shot MultiBox Detector | a PyTorch Tutorial to Object Detection ([blog](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)): a very good tutorial to refresh the details in object detection, too much details in CNN parts.
* Time Series
  * [x] Autoregressive Convolutional Neural Networks for Asynchronous Time Series ([ICML'18](http://proceedings.mlr.press/v80/binkowski18a/binkowski18a.pdf)): very hard to read (writing problem). I don't think it is good solution to treat async duration as inputs.
* Adversarial Examples
  * [x] How to know when machine learning does not know ([blog](http://www.cleverhans.io/security/2019/05/20/dknn.html))
  * [ ] Adversarial Examples Are Not Bugs, They Are Features ([arXiv](https://arxiv.org/abs/1905.02175))
  * [ ] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
* Reinforcement Learning
  * [ ] Recurrent World Models Facilitate Policy Evolution ([NIPS'18](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf)) ([website](https://worldmodels.github.io/))
  * [ ] ~Continuous control with deep reinforcement learning (a.k.a. DDPG) ([ICLR'16](https://arxiv.org/abs/1509.02971))


## 2019-07

I am excited and lucky to have my [research paper](https://onlinelibrary.wiley.com/doi/full/10.1002/cam4.2233) selected as cover article on [Cancer Medicine](https://onlinelibrary.wiley.com/toc/20457634/2019/8/7)!

One paper was accepted by ICCV'19, and another one was accpeted by Thoracic Cancer (Journal). 

### Reading
* Medical
  * [ ] End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography ([Nat Medicine](https://www.nature.com/articles/s41591-019-0447-x))
  * [x] Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study ([PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002711)): though only with radiotherapy and surgery patients, it is a good study with entensive experiments.
  * [x] Deep Learning Predicts Lung Cancer Treatment Response from Serial Medical Imaging ([Clinical Cancer Research](http://clincancerres.aacrjournals.org/content/25/11/3266)): a study very similar to what I am doing.
  * [ ] Signet Ring Cell Detection With a Semi-supervised Learning Framework ([IPMI'19](https://arxiv.org/abs/1907.03954))
* 3DV
  * [x] MeshCNN: A Network with an Edge ([SIGGRAPH'19](https://arxiv.org/abs/1809.05910)) ([proj](https://ranahanocka.github.io/MeshCNN/))
  * [ ] MeshNet: Mesh Neural Network for 3D Shape Representation ([AAAI'19]()) ([code](https://github.com/Yue-Group/MeshNet))
  * [ ] PointConv: Deep Convolutional Networks on 3D Point Clouds ([CVPR'19](https://arxiv.org/abs/1811.07246))
* Adversarial Examples
  * [ ] Adversarial Examples Are Not Bugs, They Are Features ([arXiv](https://arxiv.org/abs/1905.02175))
  * [ ] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
  * [x] Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search ([ICCV'19](https://arxiv.org/abs/1903.12269)): although with not good writing and structure, it proposed a novel and interesting adversarial attack setting towards model parameter (bit-flip), with strong enough performance.

## 2019-06
Luckily, one of my MICCAI'19 submissions has been accepted! See you in Shenzhen!

I will be attending CVPR'19 (Jun 15 - 21). See you at Long Bench, CA!

Mass without complete paper reading in depth :-(

### Study
* [x] Introduction to Biomedical Imaging 生物医学成像学导论 ([Xuetangx](http://www.xuetangx.com/courses/course-v1:UQx+BIOIMG101x+sp/courseware/a9ae78343c0f47ad91159d3b9035ea9c/))
  * [x] Week 4: MRI - it is beyond my knowledge so far... too much about signal and physics.
  * [x] Week 5: PET


## 2019-05
I gave up rushing NIPS ddl, and turn to AAAI ddl due to limited experiments. Anyway, it should be a Bayesian month. 

### Study
* [ ] Introduction to Biomedical Imaging 生物医学成像学导论 ([Xuetangx](http://www.xuetangx.com/courses/course-v1:UQx+BIOIMG101x+sp/courseware/a9ae78343c0f47ad91159d3b9035ea9c/))
  * [x] Week 3: Ultrasound
  * [ ] Week 4: MRI
  * [ ] Week 5: PET
 `
### Reading
* [x] Revisiting Batch Normalization For Practical Domain Adaptation ([ICLR'17 workshop](https://openreview.net/forum?id=Hk6dkJQFx))
* Medical
  * [x] nnU-Net: Breaking the Spell on Successful Medical Image Segmentation ([arXiv, ?MICCAI'19 submission](https://arxiv.org/abs/1904.08128)): a fantastic work, though very "engineering", it proposes an important direction with very promising results. It hard encodes many engineering tricks. I suppose it to be the MICCAI'19 best paper. 
  * [x] A large annotated medical image dataset for the development and evaluation of segmentation algorithms ([arXiv](https://arxiv.org/abs/1902.09063)): an awesome study towards medical AutoML, a Nature-level dataset preparation, maybe submit to Nature Machine Intelligence?  
  * [ ] End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography ([Nat Medicine](https://www.nature.com/articles/s41591-019-0447-x))
  * [ ] Elastic Boundary Projection for 3D Medical Imaging Segmentation ([CVPR'19](https://arxiv.org/pdf/1812.00518.pdf))
* Adversarial Examples
  * [x] Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks ([CVPR'19](https://arxiv.org/abs/1904.02884)): a very insightful work. DYP is indeed a brilliant young researcher that pushes the frontier of ADV. His paper always comes with great insights / motivations, together with a simple solution / method. 
  * [x] Improving Transferability of Adversarial Examples with Input Diversity ([ECCV'18->CVPR'19](https://arxiv.org/abs/1803.06978))
  * [ ] Adversarial Examples Are Not Bugs, They Are Features ([arXiv](https://arxiv.org/abs/1905.02175))
  * [ ] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
* 3DV
  * [x] Learning to Sample ([CVPR'19](https://arxiv.org/abs/1812.01659)): not well-writen, all the paper controls is the loss. It is not elegant, and not easy to apply in practical applications. For example, the proposed methods are non-trivial to apply in hierachical architectures, e.g., PointNet++. 
  * [ ] MeshCNN: A Network with an Edge ([SigGraph'19](https://arxiv.org/abs/1809.05910)) ([proj](https://ranahanocka.github.io/MeshCNN/))
  * [ ] MeshNet: Mesh Neural Network for 3D Shape Representation ([AAAI'19]()) ([code](https://github.com/Yue-Group/MeshNet))
  * [ ] PointConv: Deep Convolutional Networks on 3D Point Clouds ([CVPR'19](https://arxiv.org/abs/1811.07246))
* Uncertainty
  * [x] Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers ([ECCV'18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Apoorv_Vyas_Out-of-Distribution_Detection_Using_ECCV_2018_paper.pdf)): not well writen. Bad logic. Somewhat good results. However, it lacks a strong baseline: CIFAR-80 vs. CIFAR-20.
  * [x] Learning Confidence for Out-of-Distribution Detection in Neural Networks ([arXiv](https://arxiv.org/abs/1802.04865)): a simple scheme, yet effective. 
  * [x] What uncertainties do we need in bayesian deep learning for computer vision ([NIPS'17](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf)): a very insightful paper, well writen, with good empirical results. It worth NIPS. Discuss data (aleatoric) uncertainty, model (epistemic) uncertainty. 
  * [x] Predictive Uncertainty Estimation via Prior Networks ([NeurIPS'18](https://papers.nips.cc/paper/7936-predictive-uncertainty-estimation-via-prior-networks.pdf)): also insightful. Apart from data (aleatoric) uncertainty, model (epistemic) uncertainty, it discussed separately distributional uncertainty, unsually merged in model uncertainty in previous study. Poorly writen in some sections, but overall good. It also proposes a Dirichlet Prior approach over categorial distribution, which I previously have a similar idea with. The experiments are limited. Anyway, the simplex representation for categorical distribution is very intuitive. 
  * [ ] Improving Simple Models with Confidence Profiles ([NeurIPS'18](https://papers.nips.cc/paper/8231-improving-simple-models-with-confidence-profiles.pdf))
  * [ ] A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks
 ([NeurIPS'18](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks))
  * [ ] Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples ([ICLR'18](https://arxiv.org/abs/1711.09325))
  * [ ] Hierarchical Novelty Detection for Visual Object Recognition ([CVPR'18](https://arxiv.org/abs/1804.00722))

## 2019-04

A terrible month. Mass. It sounds good: I have submited papers to MICCAI'19 and ACM MM, and a paper was accepted by Cancer Medicine. However, my research have little progress indeed. Besides I spent too much time on games (especially `The Legend of Zelda: Breath of the Wild` :) ). I need focus. For further plan, I need more inputs. Only by this I can go further. Bravo, Jiancheng!

### Reading
* Graph Neural Networks
  * [x] How Powerful are Graph Neural Networks? ([ICLR'19](https://openreview.net/forum?id=ryGs6iA5Km)): very good paper, intuitive, insightful. But writing is not great enough. Maybe just because there are many "not such systematic" ideas in this paper. 
  * [x] Graph U-Net ([ICLR'19 rej->ICML'19](https://openreview.net/forum?id=HJePRoAct7)) 

## 2019-03

### Study
* Neural Process / Non-parametric
  * [x] Neural Processes as distributions over functions ([blog](https://kasparmartens.rbind.io/post/np/))
  * [x] Conditional Neural Process ([ICML'18](https://arxiv.org/pdf/1807.01613.pdf))
* Uncertainty
  * [x] Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles ([NIPS'17](https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf))
  * [x] The need for uncertainty quantification in machine-assisted medical decision making ([NatMI](https://www.nature.com/articles/s42256-018-0004-1))
  * [x] GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training ([ACCV'18](https://arxiv.org/abs/1805.06725)): poorly writen. Limited insights. Something bizarre: how can AUC be less than 0.5?
* MIC
  * [x] Transferable Multi-model Ensemble for Benign-Malignant Lung Nodule Classification on Chest CT ([MICCAI'17](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_75)): poorly writen with little novel insights.
  * [x] Diagnostic Classification Of Lung Nodules Using 3D Neural Networks ([ISBI'18](httpst://arxiv.org/abs/1803.07192))
  * [x] Semi-Supervised Multi-Task Learning for Lung Cancer Diagnosis ([arXiv](https://arxiv.org/pdf/1802.06181.pdf))
  * [x] Automated Pulmonary Nodule Detection: High Sensitivity with Few Candidates ([MICCAI'18](https://www.researchgate.net/publication/327629744_Automated_Pulmonary_Nodule_Detection_High_Sensitivity_with_Few_Candidates_21st_International_Conference_Granada_Spain_September_16-20_2018_Proceedings_Part_II))


## 2019-02
* During Spring Festival, it was a busy time for CVPR'19 rebuttal. Luckily, 1 of 3 first-author submissions of mine was accepted to CVPR'19! 
* I then traveled to Thailand and HK. It was a busy month yet with few paper reading.

### Reading
* Adversarial Robustness
  * [x] Boosting Adversarial Attacks with Momentum ([CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)): though simple, it is incretibly effective. The writing of the paper is great. Well motivated. Insightful. NIPS2017 AdvExp chanllenge winners of two tracks (untargeted and targeted attack).
  * [x] Mitigating Adversarial Effects Through Randomization ([ICLR'18](https://openreview.net/forum?id=Sk9yuql0Z)): bypassed by Obfuscated Gradient paper. Good to read the challenge setting.
* [x] Learning to learn by gradient descent by gradient descent ([NIPS'16](https://arxiv.org/abs/1606.04474)): good insight on learning to learn. Besides, new insights for me, apart from the models and algorithms, are that we can use any value for backprop! Even for a value that is not in a same computation graph! It's actually very straighforward, but very difficult to get it for the first time.
* [x] Do Deep Generative Models Know What They Don't Know? ([ICLR'19](https://openreview.net/forum?id=H1xwNhCcYm)): OoD samples can have higher likelihood for generative models (emphasis on flow). However, it seems that anomaly detection is still applicable?

## 2019-01

### Study
* [ ] Introduction to Biomedical Imaging 生物医学成像学导论 ([Xuetangx](http://www.xuetangx.com/courses/course-v1:UQx+BIOIMG101x+sp/courseware/a9ae78343c0f47ad91159d3b9035ea9c/))
  * [ ] Week 3: Ultrasound

### Reading
* [x] A Probabilistic U-Net for Segmentation of Ambiguous Images ([NIPS'18](https://arxiv.org/abs/1806.05034)) (code: [official re-implementation (TensorFlow)](https://github.com/SimonKohl/probabilistic_unet), [PyTorch](https://github.com/stefanknegt/probabilistic_unet_pytorch))
* [x] Snapshot Ensembles: Train 1, get M for free ([ICLR'17](https://arxiv.org/abs/1704.00109)): not so effective, but consine learning rate seems potential. 
* [x] Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights ([ECCV'18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf)): very inspiring.
* Videos
  * [x] Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset ([CVPR'17](https://arxiv.org/abs/1705.07750)) ([code](https://github.com/deepmind/kinetics-i3d)): a.k.a. I3D
  * [x] Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? ([CVPR'18](https://arxiv.org/abs/1711.09577)) ([code](https://github.com/kenshohara/3D-ResNets-PyTorch)): seems very limited novelty, but may be good to use.
* Dynamic Networks
  * [x] DARTS: Differentiable Architecture Search ([ICLR'19](https://arxiv.org/abs/1806.09055)) ([OpenReview](https://openreview.net/forum?id=S1eYHoC5FX) disputed): interesting paper, inspiring, validation dataset is also possible for (iteractive) optimization.
  * [x] Efficient Neural Architecture Search via Parameter Sharing ([arXiv](https://arxiv.org/abs/1802.03268)): a.k.a. ENAS. 
    * *NB*: Read DARTS and ENAS, it's easy to find these two are highly related. Lots of contributions are proposed by ENAS (or even before?). The experiment settings of DARTS are borrowed much from ENAS. A basic assumption of DARTS is the weight sharing (which is the key contribution of ENAS). DARTS provides an alternative to optimize the networks: ENAS uses RL (with controller LSTM), and DARTS use a soft weights. A single forward for ENAS is just a network, while for DARTS it is an "ensemble" of networks. For experiments, ENAS seems more efficient, though the controller LSTM get gradients with high variance. For me, I prefer ENAS way (just like neural turing machine), while DARTS is clearly easier to implement. The [Switchable Norm](https://arxiv.org/abs/1806.10779) is inspired from / could be viewed as variant of DARTS.  
  * [x] Decoupled Neural Interfaces Using Synthetic Gradients ([ICML17](http://proceedings.mlr.press/v70/jaderberg17a/jaderberg17a.pdf)) ([DeepMind blog](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/)): the blog is great for illustration.
  * [x] A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play ([Science](http://science.sciencemag.org/content/362/6419/1140)): many details missing in the context. AlphaGO, AlphaGO Zero paper and supplementary are also needed to implement (though it is so large a project...). For example, the networks are presented in AlphaGO Zero paper. The reward is not detailed in AlphaZero paper as well.
  * [x] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context ([ICLR'19 rejected](https://arxiv.org/abs/1901.02860)): on language modeling, not on seq2seq task. Some "memory"-like improvements. Impressive results, but seems not easy to transfer to more realistic seq2seq task (like NMT). 
  * [x] GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations ([NIPS'18]): transfer learned graph (element relations) only, rather than directly transfer features / networks. While it seems very tricky instead.
  * [x] Watch Your Step: Learning Node Embeddings via Graph Attention ([NIPS'18](https://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention.pdf)): use attention in embedding learning, rather than inference, to dynamically determine the feature horizon. I missed some details on graph embeddings.
  * [x] Slimmable Neural Networks ([ICLR'19](https://openreview.net/forum?id=H1gMCsAqY7))
  * [x] Hierarchical Graph Representation Learning with Differentiable Pooling ([NIPS'18](http://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling))
* 3DV
  * [x] Learning Category-Specific Mesh Reconstruction from Image Collections ([ECCV18](https://arxiv.org/abs/1803.07549)) ([project page](https://akanazawa.github.io/cmr/))
  * [x] Realistic Adversarial Examples in 3D Meshes ([OpenReview](https://openreview.net/forum?id=SJfcrn0qKX): ICLR19 withdraw): it is indeed very creative, while it was done in a rush. Pitfalls: single-viewpoint issues and the paper writing.
  * [x] Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects ([arXiv](https://arxiv.org/abs/1811.11553))
  * [x] Unsupervised Learning of Shape and Pose with Differentiable Point Clouds ([NIPS18](https://arxiv.org/abs/1810.09381)): Very interesting paper, it is not only a point cloud version of neural mesh renderer! It also talks about the camera pose estimation, with joint training, and an ensemble distillation without a REINFORCE-based method able to optimize! Very interesting!
