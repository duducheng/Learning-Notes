# Learning Notes by [Jiancheng Yang](https://jiancheng-yang.com/)

**Update 2021-04-13**: I will no longer update my reading list regularly, since now many reading materials of mine are not good to share publicly (e.g., conference / journal reviews). Besides, I only skim papers in most cases now, which makes it annoying to record everything using git. However, I will still share thoughts on some really interesting papers. 

Archives: [2019](archive/note19.md) | [2018](archive/note18.md) | [before 2018](archive/before18.md)

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
* [ ] Alleviating Class-wise Gradient Imbalance for Pulmonary Airway Segmentation ([arXiv](https://arxiv.org/abs/2011.11952))
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
