NB: Time below means when I "studied", not when it came.

### 2017-03 Week 4
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ([arXiv](https://arxiv.org/abs/1502.03167)): simple math but full of brilliant ideas and tricks. ([code](https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412): good demenstration of using "global" moment and `ewa`)
* Layer Normalization ([arXiv](https://arxiv.org/abs/1607.06450)): even simpler principle, good for RNN but, worse than BN for CNN.
* [Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](notes/plstm.md) ([NIPS](https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf)) (TensorFlow [implement](https://github.com/Enny1991/PLSTM), Keras [implement](https://github.com/fferroni/PhasedLSTM-Keras), both good and clear.)
* Using Fast Weights to Attend to the Recent Past ([arXiv](https://arxiv.org/abs/1610.06258)) (TensorFlow [implement](https://github.com/ajarai/fast-weights)): Very simple math, and easy enough to implement, but it seems lots of physiology background. This paper is another trial aimed to beat LSTM. Fast weights (`FW`) based on IRNN, works well on the mentioned task. The `FW` can be regarded as something to be "memorised" during the step update. I found papers of Hinton are usually recondite. (maybe Canadian English?)
* Neural Networks for Machine Learning by Geoffrey Hinton ([Coursera](https://www.coursera.org/learn/neural-networks/)): Finally finished. Good review for neural network approaches. Absolutely not a first course. It's a very course that can inspire you a lot if you've already known; but if you haven't known something mentioned in the course, it can be very hard for you to fully understand without other materials.
* [MLaPP](https://www.cs.ubc.ca/~murphyk/MLbook/): Chapter 27.7 Restricted Boltzman machines (RBMs)

## 2017-03
* MyWeekly
  * [Personal Proceeding on Time Series: DTW, Viz of RNN and Clockwalk RNN Revisiting](weekly/dtw_vizrnn_cwrnn.pdf)
  * [Personal Proceeding on Time Series (2): Echo State Network and Temporal Kernel RNN](weekly/tkrnn_esn.pdf)
  * [Personal Proceeding on Time Series (3): Phased LSTM and STL](weekly/plstm_stl.pdf)
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
