# A quick overview of recent Auto Augmentation techniques
The AutoAug (AA) [1] is originally a RL-based controller of augmentation policy search, which is very computational entensive. There are lots of improvements over AA, however they uses very different search phases for augmentation policies, which makes them difficult to be compared with each other indeed.

| Paper | Parameterization of Augmentation Policies | Search Phase |
|:---:|:---:|:---:|
| Fast AA  [2]| Non-Diff Aug + Bayesian Opt | K-Fold; Training from scrath without aug; Evaluate policy path with finetune |
| Adv AA [3] | Non-Diff Aug + RL controller | No separate search phase; adversarial policy generator to gradient ascent classification loss |
| Faster AA [4] | Diff Aug (Gumbel-Softmax for Selection and Straight-Through for Magnitudes) | GAN training to make aug policies looking like the training data |
| DATA [5] | Diff Aug (RELAX for Selection and Straight-Through for Magnitudes) | DARTS-style one-pass training and validation |
| RandAugment [6] | Only control the number of policy used (`N`) and a unifed magnitude (`M`) for all policies | Grid search of `(N, M)`|
Note: RELAX gradient estimator is just an improved Gumbel Softmax with unbiased gradient. Faster AA and DATA looks very similar except for the search phase.

Reference:
1. AutoAugment: Learning Augmentation Policies from Data ([CVPR'19](https://arxiv.org/abs/1805.09501))
2. Fast AutoAugment ([NeurIPS'19](https://papers.nips.cc/paper/2019/hash/6add07cf50424b14fdf649da87843d01-Abstract.html))
3. Adversarial AutoAugment ([ICLR'20](https://arxiv.org/abs/1912.11188))
4. Faster AutoAugment: Learning Augmentation Strategies Using Backpropagation ([ECCV'20](https://www.ecva.net/papers/* eccv_2020/papers_ECCV/papers/123700001.pdf))
5. DADA: Differentiable Automatic Data Augmentation ([ECCV'20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670579.pdf))
6. RandAugment: Practical automated data augmentation with a reduced search space ([NeurIPS'20](https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf))
