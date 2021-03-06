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