## Study Plan
AML - [Advanced Machine Learning](topics/advanced_ml.md) | MIC - [Medical Image Computing](topics/mic.md) | Prog - [Programming](topics/programming.md)

Related projects: 
* [`M3DV/ai-deadlines`](https://github.com/M3DV/ai-deadlines): Top AI deadline countdowns (with an emphasis on computer vision and medical images).
* [`M3DV/Readiness-Seminar`](https://github.com/M3DV/Readiness-Seminar): A cooperative paper list on 3D vision and deep learning robustness.
* [`seanywang0408/Kickstart`](https://github.com/seanywang0408/Kickstart): Study route for learners in machine learning / deep learning / computer vision.

NB: Time below means when "I am / was studying", not that for the paper itself.


## 2019-10
I have just submitted papers to ISBI 2019 and will be attending MICCAI 2019 in Shenzhen, China.

### Reading

* Medical (Emphasis on MICCAI'19)
  * [x] Overfitting of neural nets under class imbalance: Analysis and improvements for segmentation ([MICCAI'19](https://arxiv.org/abs/1907.10982)): very insightful. Less data contributes to low sensitivity / recall to foreground rather than precision. 
  * [ ] DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation ([MICCAI'15](https://link.springer.com/chapter/10.1007%2F978-3-319-24553-9_68)) ([dataset](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT): a.k.a. NIH Pancreas CT)
  * [ ] Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation ([CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.pdf)) ([code](https://github.com/twni2016/OrganSegRSTN_PyTorch))
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
* Adversarial Examples
  * [ ] Adversarial Examples Are Not Bugs, They Are Features ([arXiv](https://arxiv.org/abs/1905.02175))
  * [ ] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations ([ICLR'19](https://arxiv.org/abs/1903.12261))
  * [ ] Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors ([ICLR'19](https://openreview.net/forum?id=BkMiWhR5K7))
  * [ ] Improving Black-box Adversarial Attacks with a Transfer-based Prior ([NeurIPS'19](https://arxiv.org/abs/1906.06919)) ([code](https://github.com/prior-guided-rgf/Prior-Guided-RGF))
* Reinforcement Learning
  * [ ] Recurrent World Models Facilitate Policy Evolution ([NIPS'18](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf)) ([website](https://worldmodels.github.io/))

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

## 2017-12
### Study Plan
* [ ] CS231n 2017 ([AML](topics/advanced_ml.md))
  * [ ] Lecture 12: Visualizing and Understanding ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf))
  * [x] Lecture 13: Generative Models ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf))
  * [x] Lecture 14: Deep Reinforcement Learning ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf))
  * [x] Lecture 15: Efficient Methods and Hardware for Deep Learning ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf))
  * [ ] Lecture 16: Adversarial Examples and Adversarial Training ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture16.pdf))
* [ ] Fundamentals of Medical Imaging ([MIC](topics/mic.md))
  * [ ] Chapter 1 
  * [ ] Chapter 2 X-rays
* [x] 生物医学成像学导论 ([MIC](topics/mic.md))
  * [x] Module 1 X-rays
### Reading
* An Overview of Multi-Task Learning in Deep Neural Networks ([blog](http://ruder.io/multi-task/)): Hard to say very good review, but mentioned a lot. **"Recent approaches have thus looked towards learning what to share"**.
* Deep Learning: Practice and Trends (NIPS 2017 Tutorial) ([YouTube](https://www.youtube.com/watch?v=YJnddoa8sHk)) ([slide](https://docs.google.com/presentation/d/e/2PACX-1vQMZsWfjjLLz_wi8iaMxHKawuTkdqeA3Gw00wy5dBHLhAkuLEvhB7k-4LcO5RQEVFzZXfS6ByABaRr4/pub?slide=id.p))
* Towards automatic pulmonary nodule management in lung cancer screening with deep learning ([Scientific Reports](https://www.nature.com/articles/srep46479))
* Special Deep Learning Structure of [MLDS](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html)
  * Spatial Transformer Layer ([slide](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2017/Lecture/Special%20Structure%20(v6).pdf))
  * Highway Network & Grid LSTM ([slide](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2017/Lecture/Special%20Structure%20(v6).pdf))


## 2017-11
* Time-series Extreme Event Forecasting with Neural Networks at Uber ([ICML Time Series Workshop](http://roseyu.com/time-series-workshop/submissions/TSW2017_paper_3.pdf))
* Deep Forecast: Deep Learning-based Spatio-Temporal Forecasting ([arXiv](https://arxiv.org/abs/1707.08110)): ICML Time Series Workshop, yet very simple work
* Neural Turing Machines ([arXiv](https://arxiv.org/abs/1410.5401)): very inspirational
* Dynamic Routing Between Capsules ([arXiv](https://arxiv.org/abs/1710.09829))
* Show, Attend and Tell: Neural Image Caption Generation with Visual Attention ([arXiv](https://arxiv.org/abs/1502.03044))
* Computerized detection of lung nodules through radiomics ([Medical Physics](http://onlinelibrary.wiley.com/doi/10.1002/mp.12331/abstract;jsessionid=749B7153927F5A6F42C3AE5BF9CF3B62.f04t01))

## 2017-10
* MyWeekly
  * [Modern CNN Design](weekly/Modern_CNN_Design.pdf)
* ["天池医疗AI大赛[第一季]：肺部结节智能诊断"](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.4c5fd3b5Y0gzo&raceId=231601) (Team: `LAB518-CreedAI`, rank: 3 / 2887)
* Identity Mappings in Deep Residual Networks (a.k.a. ResNet-1001 / ResNet200) ([arXiv](https://arxiv.org/abs/1603.05027))
* ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices ([arXiv](https://arxiv.org/abs/1707.01083))
* Squeeze-and-Excitation Networks ([arXiv](https://arxiv.org/abs/1709.01507))
* Deep Convolutional Neural Networks with Merge-and-Run Mappings ([arXiv](https://arxiv.org/abs/1611.07718))
* Interleaved Group Convolutions for Deep Neural Networks ([arXiv](https://arxiv.org/abs/1707.02725))

## 2017-09
* WSISA: Making Survival Prediction from Whole Slide Histopathological Images ([CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_WSISA_Making_Survival_CVPR_2017_paper.pdf))
  - *NB*: it's suitable for small data cases while large data size (large per case). It use KMeans in the paper for patch level clustering, train separate models and use the prediction as features. Not beautiful solution.
* Feature Pyramid Networks for Object Detection ([arXiv](https://arxiv.org/abs/1612.03144))
  - *NB*: clean top results with single beautiful. Some so-called bells and whistles that have not been tried in the paper: iterative regression [9], hard negative mining [35], context modeling [16], strong data augmentation [22], etc.
* Xception: Deep Learning with Depthwise Separable Convolutions ([arXiv](https://arxiv.org/abs/1610.02357))
* Aggregated Residual Transformations for Deep Neural Networks (a.k.a. ResNeXt) ([arXiv](https://arxiv.org/abs/1611.05431))
  * *NB*: Group Conv, interpreted as "Network in Neuron".

## 2017-08
* Internship in Tencent Social Ads Team @Shenzhen
* Show and Tell: A Neural Image Caption Generator ([arXiv](https://arxiv.org/abs/1411.4555))
* Dual Path Networks ([arXiv](https://arxiv.org/abs/1707.01629))
* Densely Connected Convolutional Networks ([arXiv](https://arxiv.org/abs/1608.06993))
* VoxResNet: Deep Voxelwise Residual Networks for Volumetric Brain Segmentation ([arXiv](https://arxiv.org/abs/1608.05895)) ([project page](http://appsrv.cse.cuhk.edu.hk/~hchen/research/seg_brain.html))

## 2017-07
* Busy month for ["天池医疗AI大赛[第一季]：肺部结节智能诊断"](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.4c5fd3b5Y0gzo&raceId=231601) (Team: `LAB518-CreedAI`, Season 1 rank: 5 / 2887)
* Conditional Random Fields as Recurrent Neural Networks ([arXiv](https://arxiv.org/abs/1502.03240))
* Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer ([arXiv](https://arxiv.org/abs/1701.06538))
* Accurate Pulmonary Nodule Detection in Computed Tomography Images Using Deep Convolutional Neural Networks ([arXiv](https://arxiv.org/abs/1706.04303))
* Dilated Residual Networks ([arXiv](https://arxiv.org/abs/1705.09914))
* Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour ([arXiv](https://arxiv.org/abs/1706.02677))

## 2017-06
* Multilevel Contextual 3-D CNNs for False Positive Reduction in Pulmonary Nodule Detection ([IEEE](http://ieeexplore.ieee.org/document/7576695/))
* Wide Residual Networks ([arXiv](https://arxiv.org/abs/1605.07146))
* D. Silver Lecture 1: Introduction to Reinforcement Learning ([UCL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html))
* CS231n ([Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html)): Lecture 2 (Linear classification I), Lecture 3 (Linear classification II), Lecture 4 (Backpropagation), Lecture 5 (Training 1), Lecture 6 (Training 2), Lecture 7 (ConvNets), Lecture 10 (RNN). 
  * Till now, finished all CS231n lectures (vidoes), notes and some readings.
* DL book RNN chapter ([link](http://www.deeplearningbook.org/contents/rnn.html))

## 2017-05
* SSD: Single Shot MultiBox Detector ([arXiv](https://arxiv.org/abs/1512.02325))
* Attention and Augmented Recurrent Neural Networks ([Distill](http://distill.pub/2016/augmented-rnns/))
* R-FCN: Object Detection via Region-based Fully Convolutional Networks ([arXiv](https://arxiv.org/abs/1605.06409))
* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks ([arXiv](https://arxiv.org/abs/1506.01497))
* Training Region-based Object Detectors with Online Hard Example Mining ([arXiv](https://arxiv.org/abs/1604.03540))
* Deep Learning, NLP, and Representations ([blog](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/))
* CS231n ([Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html)): Lecture 12 (Deep Learning Tools), 14 (Videos, Unsupervised learning), 15 (Invited Talk: Jeff Dean)

## 2017-04
* MyWeekly
  * [Restricted Boltzmann Machines](weekly/rbm.pdf)
* Recurrent Dropout without Memory Loss ([arXiv](https://arxiv.org/abs/1603.05118))
  * *NB*: simple, implemented in TensorFlow. It archieves similar (if not better) results in LSTMs than Gal 2015. In short, it drops the recurrent updates but not the recurrent connections, it allows per-step dropout. Moon et al. 2015 drop the recurrent update and connections, with per-sequence dropout, which allows long-term learning but forget the long-term memory in inference.
* CS231n ([Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html)): Lecture 8 (Detection), 13 (Segmentation and Attention)
* The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation ([arXiv](https://arxiv.org/abs/1611.09326)): there are some typos in Table2.
* Multi-Scale Context Aggregation by Dilated Convolutions ([arXiv](https://arxiv.org/abs/1511.07122)), a.k.a "Dilated-8"
* A Simple Way to Initialize Recurrent Networks of Rectified Linear Units ([arXiv](https://arxiv.org/abs/1504.00941)), a.k.a "IRNN"
* Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting ([arXiv](https://arxiv.org/abs/1506.04214))
* Understanding Convolutions ([blog](http://colah.github.io/posts/2014-07-Understanding-Convolutions/))
* Groups & Group Convolutions ([blog](http://colah.github.io/posts/2014-12-Groups-Convolution/))
* Deconvolution and Checkerboard Artifacts ([Distill](http://distill.pub/2016/deconv-checkerboard/))
* Calculus on Computational Graphs: Backpropagation ([blog](http://colah.github.io/posts/2015-08-Backprop/))
* Neural Networks, Manifolds, and Topology ([blog](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/))

## 2017-03
* MyWeekly
  * [On Time Series: DTW, Viz of RNN and Clockwalk RNN Revisiting](weekly/dtw_vizrnn_cwrnn.pdf)
  * [On Time Series (2): Echo State Network and Temporal Kernel RNN](weekly/tkrnn_esn.pdf)
  * [On Time Series (3): Phased LSTM and STL](weekly/plstm_stl.pdf)
  * [On Time Series (4): Fast weights](weekly/fast_weights.pdf)
* [STL: A Seasonal-Trend Decomposition Procedurue Base on Loess](notes/stl.md) ([link](http://www.wessa.net/download/stl.pdf))
* [Temporal-Kernel Recurrent Neural Networks](notes/tkrnn.md) ([ScienceDirect](http://www.sciencedirect.com/science/article/pii/S0893608009002664))
* [REVISIT] A Clockwork RNN ([arXiv](https://arxiv.org/abs/1402.3511)) (non-official [code](https://github.com/braingineer/ikelos/blob/master/ikelos/layers/cwrnn.py))
* [Visualizing and Understanding Recurrent Networks](notes/viz_rnn.md) ([arXiv](https://arxiv.org/abs/1506.02078))
* Neural Networks for Time Series Prediction ([CMU](https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/slides/timeseries.pdf)): super old lecture, even not covering LSTM. While still useful, especially it talks many concepts of time series analysis in engineering guys' eyes (rather than statstician's), though some of them are too "Digital Signal Processing" that make my undergraduate "Signal & System" concepts revive :)
* Dynamic Time Wrapping
  * *NB*: Yet another example of dynamic programming in sequence modeling, I think CTC's idea benifits from DTW (and absolutely HMM).
  * K Nearest Neighbors & Dynamic Time Warping ([code](https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping)): clean code, using DTW and kNN for Human Activity Recognition. It clearly shows the esential idea of DTW, and the code is well factored. But something funny is that, in this code, not all the imports are valid, you should import something manualy before running the code.
  * Everything you know about Dynamic Time Warping is Wrong ([link](http://wearables.cc.gatech.edu/paper_of_week/DTW_myths.pdf)): gives some highlights of using and researching DTW (about 10 years ago 😐). The wording of this paper is very sharp. 3 chaims: 1) fix length doesn't hurt 2) narrow band doesn't hurt 3) speeding up DTW with tight lower bound is pointless.
* MC and MCMC from Probabilistic Graphical Models Eric Xing ([CMU](http://www.cs.cmu.edu/~epxing/Class/10708-14/lecture.html)): Lecture 16-18.
  * *NB*: great review for sampling based inference. MC: naive, rejection sampling, importance sampling. MCMC: Metropolis-Hasting, Gibbs, collapsed (Rao-Blackwellised) Gibbs, slice sampling, Reversible Jump MCMC (RJMCMC). RJMCMC is really non-trivial, which I didn't fully understand. It's a MCMC to jump among models' space, designed without detailed balance, while stationary.
* Probabilistic Programming & Bayesian Methods for Hackers ([link](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)) ([code](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers))
* Probabilistic Graphical Models 3: Learning ([Coursera](https://www.coursera.org/learn/probabilistic-graphical-models-3-learning/))
* [Forecasting at Scale](notes/prophet.md) ([Prophet](https://facebookincubator.github.io/prophet/))
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ([arXiv](https://arxiv.org/abs/1502.03167)): simple math but full of brilliant ideas and tricks. ([code](https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412): good demenstration of using "global" moment and `ewa`)
* Layer Normalization ([arXiv](https://arxiv.org/abs/1607.06450)): even simpler principle, good for RNN but, worse than BN for CNN.
* [Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](notes/plstm.md) ([NIPS](https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf)) (TensorFlow [implement](https://github.com/Enny1991/PLSTM), Keras [implement](https://github.com/fferroni/PhasedLSTM-Keras), both good and clear.)
* Using Fast Weights to Attend to the Recent Past ([arXiv](https://arxiv.org/abs/1610.06258)) (TensorFlow [implement](https://github.com/ajarai/fast-weights)): Very simple math, and easy enough to implement, but it seems lots of physiology background. This paper is another trial aimed to beat LSTM. Fast weights (`FW`) based on IRNN, works well on the mentioned task. The `FW` can be regarded as something to be "memorised" during the step update. I found papers of Hinton are usually recondite. (maybe Canadian English?)
* Neural Networks for Machine Learning by Geoffrey Hinton ([Coursera](https://www.coursera.org/learn/neural-networks/)): Finally finished. Good review for neural network approaches. Absolutely not a first course. It's a very course that can inspire you a lot if you've already known; but if you haven't known something mentioned in the course, it can be very hard for you to fully understand without other materials.
* [MLaPP](https://www.cs.ubc.ca/~murphyk/MLbook/): Chapter 27.7 Restricted Boltzman machines (RBMs)

## 2017-02
Reprise from the Spring Festival 😐
* A Critical Review of Recurrent Neural Networks for Sequence Learning ([arXiv](https://arxiv.org/abs/1506.00019))
  * *NB*: there is not any new insight, while good to reflash some idea; it talks about vanilla RNN, LSTM, BRNN and a little bit NTM, and introduce some application, with emphasis on NLP.
* [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](notes/ctc.md) ([ICML](http://www.cs.toronto.edu/~graves/icml_2006.pdf))
* Bootstrap Methods for Time Series ([link](https://www.kevinsheppard.com/images/0/0a/Kreiss_and_lahiri.pdf))
  * *NB*: Though the article regards itself as a simple intro, it seems too theoretical for me. It provides some good review of bootstrap for time series, remember 1. generate series for AR like model; 2. block bootstrap; 3. markov chain bootstrap; 4. frequency domain by DFT 5. other mixtures.
* Hidden Markov Model
  * Markov model and HMM by mathematicalmonk ([YouTube](https://www.youtube.com/watch?v=7KGdE2AK_MQ&list=PLD0F06AA0D2E8FFBA&index=95)): covering forward-backward and Viterbi.
  * [统计学习方法](https://book.douban.com/subject/10590856/) 第10章 隐马尔可夫模型
* Conditional Random Field
  * Lecture from CMU ([YouTube](https://www.youtube.com/watch?v=B1nl8fLgKMk&t=1748s))
  * [统计学习方法](https://book.douban.com/subject/10590856/) 第11章 条件随机场
* Monte Carlo by mathematicalmonk ([YouTube](https://www.youtube.com/watch?v=AadKNJU1-lk&list=PLD0F06AA0D2E8FFBA&index=1275)): covering importance sampling, Smirnov transform and rejection sampling.
* MCMC by mathematicalmonk ([YouTube](https://www.youtube.com/watch?v=7KGdE2AK_MQ&list=PLD0F06AA0D2E8FFBA&index=95)): covering ergodic theorem and Metropolis, very gentle and intuitive.
* Probabilistic Programming & Bayesian Methods for Hackers ([code](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers))
  * *NB*: Great book, about practical Bayesian modeling and PyMC3.
* Variation Inference from Probabilistic Graphical Models Eric Xing ([CMU](http://www.cs.cmu.edu/~epxing/Class/10708-14/lecture.html)): Lecture 13-15.
  * *NB*: really good (somehow advanced) introduction to VI: loopy belief Propagation, mean field approximation, and general variational principles (solve problem in optimization fashion with dual form of function). The principles part is really abstract, but at least I got the idea. A brief to LDA is also presented in Lecture 15.
* Bayesian optimization by Nando de Freitas ([YouTube](https://www.youtube.com/watch?v=vz3D36VXefI))
  * *NB*: Great intro to Bayesian opt, in 10 minutes you get the whole picture, and the rest tells some details.

## 2017-01
* MyWeekly
  * [Advanced Linear Regression](weekly/Advanced_Linear_Regression.pdf)
* Online Kernel Ridge
  * Online regression with kernel ([link](https://gtas.unican.es/files/pub/ch21_online_regression_with_kernels.pdf))
    * *NB*: Indeed a good review for this family of approach, which really hits my ideas (how HAVE they "copied" my idea T_T). However, this paper put strong emphasis on Signal Processing, which is not suit for Machine Learning mind.
  * Local online kernel ridge regression for forecasting of urban travel times ([ScienceDirect](http://www.sciencedirect.com/science/article/pii/S0968090X14001648))
* Gaussian Processes: A Quick Introduction ([arXiv](https://arxiv.org/abs/1505.02965))
* CS229 Lecture note of Gaussian Process ([Stanford](http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf))
* Bayesian Linear Regression ([YouTube](https://www.youtube.com/watch?v=dtkGq9tdYcI)): A very good deduction, while use not commonly used symbols.
* Recursive Least Square (RLS) ([OTexts](https://www.otexts.org/1582))
* [Time Series Blogs by QuantStart](https://www.quantstart.com/articles/#time-series-analysis)
  * *NB*: These blogs are quite good, covering lots of concepts on Time Series Analysis. While it focus on the representation level, not on the learning level, which means there is less content in these blogs talking about parameter estimate. Anyway, good introduction to ARIMA, GARCH, Kalman Filter and HMM on Time Series.
* Understanding the Basis of the Kalman Filter Via a Simple and Intuitive Derivation ([IEEE](http://ieeexplore.ieee.org/document/6279585/)): Lecture note. It's quite intuitive to understand the "basis" of Kalman Filter, as titled.
* [MLaPP](https://www.cs.ubc.ca/~murphyk/MLbook/): Chapter 18 State Space Model
* Probabilistic Graphical Models 2: Inference ([Coursera](https://www.coursera.org/learn/probabilistic-graphical-models-2-inference/))
* [Probabilistic Graphical Models: Principles and Techniques](http://pgm.stanford.edu/): Chapter 9, 10, 11, 13 (selected sections)

## 2016-12
* Spectral Clustering
  * On Spectral Clustering: Analysis and an algorithm ([NIPS](http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf))
  * Spectral Clustering Tutorial Series by Chieu from NEU ([YouTube](https://www.youtube.com/playlist?list=PLdk2fd27CQzT7opzoGHvqDuDbltozWn7O)): Good but too trivial
  * Implement the Gaussian Mixture Models ([notebooks](https://github.com/duducheng/clustering))
* [统计学习方法](https://book.douban.com/subject/10590856/) 第9章 EM算法及其推广

## 2016-11

* MyWeekly
  * [Recipe of Gradient Descent](weekly/Recipe_Gradient_Descent.pdf)
  * [Brief Intro to Variational Inference (Minimal Statistics Background)](weekly/Intro_Variational_Inference.pdf)
* Optimization on Neural Networks
  * How to make the learning go faster by Geoffrey Hinton ([Coursera](https://www.coursera.org/learn/neural-networks/): Week 6)
  * An overview of gradient descent optimization algorithms ([blog](http://sebastianruder.com/optimizing-gradient-descent/))
  * CS231n Course Notes on gradient based optimization ([Stanford](http://cs231n.github.io/neural-networks-3/))
* [REVISIT] Convolution for Neural Networks
  * Convolution arithmetic tutorial ([blog](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic))
  * Conv Nets: A Modular Perspective ([blog](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/))
* Probabilistic Graphical Models 1: Representation ([Coursera](https://www.coursera.org/learn/probabilistic-graphical-models/))
* [Probabilistic Graphical Models: Principles and Techniques](http://pgm.stanford.edu/): Chapter 1, 2, 3, 5, 6
  * *NB*: I read the Chinese Version ([概率图模型：原理与技术](http://www.tup.tsinghua.edu.cn/bookscenter/book_03992101.html)), quite good if you are taking the course (and you are Chinese of course); if not, there will be something confusing in the translation version. Anyway, great thanks to the effort of Prof. Wang and Prof. Han
* Implement the Gaussian Mixture Models ([code](https://github.com/rushter/MLAlgorithms/blob/master/mla/gaussian_mixture.py)) ([notebooks](https://github.com/duducheng/clustering))
* Visual Information Theory ([blog](https://colah.github.io/posts/2015-09-Visual-Information/))
* Towards my research
  * Two Machine Learning Approaches for Short-Term Wind Speed Time-Series Prediction ([IEEE](http://ieeexplore.ieee.org/document/7091914/))
  * Forecasting day ahead electricity spot prices: The impact of the EXAA to other European electricity markets ([arXiv](https://arxiv.org/abs/1501.00818))



## 2016-10
* MyWeekly
  * [More on RNNs](weekly/More_on_RNNs.pdf)
  * [Thoughts on Wavenet](weekly/Thoughts_on_Wavenet.pdf)
  * [RNN and LSTM](weekly/RNN-LSTM.pdf)
  * [Attention](weekly/Attention.pdf)
* Topics on RNN
  * [REVISIT] Understanding LSTM Networks ([blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)) ([code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/6_lstm.ipynb))
    - *NB*: The code from Udacity Deep Learning course is exactly the same as described in the blog.
  * LSTM: A Search Space Odyssey ([arXiv](https://arxiv.org/abs/1503.04069))
  * An Empirical Exploration of Recurrent Network Architectures ([JMLR](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf))
  * Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling ([arXiv](https://arxiv.org/abs/1412.3555))
  * A Theoretically Grounded Application of Dropout in Recurrent Neural Networks ([arXiv](https://arxiv.org/abs/1512.05287))
  * Sequence to Sequence Learning with Neural Networks ([arXiv](https://arxiv.org/abs/1409.3215))
  * A Clockwork RNN ([arXiv](https://arxiv.org/abs/1402.3511))
  * Recurrent Neural Networks for Multivariate Time Series with Missing Values ([arXiv](https://arxiv.org/abs/1606.01865))
  * Deep Learning Lecture 13: Alex Graves on Hallucination with RNNs ([YouTube](https://www.youtube.com/watch?v=-yX1SYeDHbg))
  * RNNs in TensorFlow, a Practical Guide and Undocumented Features ([blog](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/))
  * Recurrent Neural Network Regularization ([arXiv](https://arxiv.org/abs/1409.2329))
  * Multi-task Sequence to Sequence Learning ([arXiv](https://arxiv.org/abs/1511.06114))
* Topics on Variational Autoencoders
  * Variational Inference Tutorial Series by Chieu from NEU ([YouTube](https://www.youtube.com/playlist?list=PLdk2fd27CQzSd1sQ3kBYL4vtv6GjXvPsE))
  * Deep Learning Lecture 14: Karol Gregor on Variational Autoencoders and Image Generation ([YouTube](https://www.youtube.com/watch?v=P78QYjWh5sM))
  * Building Autoencoders in Keras ([blog](https://blog.keras.io/building-autoencoders-in-keras.html))
  * Markov Chain Monte Carlo Without all the Bullshit ([blog](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/))
* [WaveNet: A Generative Model for Raw Audio](notes/wavenet.md)  ([arXiv](https://arxiv.org/abs/1609.03499)) ([blog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)) ([code](https://github.com/usernaamee/keras-wavenet))
  - *NB*: The repo of code is a very simple version to read.
* t-SNE
  * Visualizing Data Using t-SNE ([YouTube](https://www.youtube.com/watch?v=RJVL80Gg3lA))
  * How to Use t-SNE Effectively ([blog](http://distill.pub/2016/misread-tsne/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue))
* [Neural Style and Deep Dream](notes/neural-style-deep-dream.md)
  * A Neural Algorithm of Artistic Style ([arXiv](https://arxiv.org/abs/1508.06576))
  * Keras Deep Dream ([code](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py))
  * Keras Neural Style ([code](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py))
  * CS231n ([Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html)): Lecture 9
* Deep Learning for ChatBots (blog: [Part1](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/) [Part2](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/))
* Courses
  * CS224d ([Stanford](http://cs224d.stanford.edu/syllabus.html)): Lecture 1 - 3
  * CS231n ([Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html)): Lecture 1
  * Neural Networks for Machine Learning by Geoffrey Hinton ([Coursera](https://www.coursera.org/learn/neural-networks/)): Week 1 - 3
* Towards my research
  * A Novel Empirical Mode Decomposition With Support Vector Regression for Wind Speed Forecasting ([IEEE](http://ieeexplore.ieee.org/document/6895279/))

## 2016-09
* [Spatial Transformer Networks](notes/spatial-transformer-network.md) ([arXiv](https://arxiv.org/abs/1506.02025)) ([blog](http://torch.ch/blog/2015/09/07/spatial_transformers.html)) ([code](https://github.com/tensorflow/models/tree/master/transformer))
* Attention and Memory in Deep Learning and NLP ([blog](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/))
* Deep Residual Learning for Image Recognition ([arXiv](http://arxiv.org/abs/1512.03385))
* Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning ([arXiv](https://arxiv.org/abs/1602.07261))
* Bilinear CNN Models for Fine-grained Visual Recognition ([arXiv](https://arxiv.org/abs/1504.07889))
