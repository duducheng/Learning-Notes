# [CTC](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

Notes:
* This paper shows the usage of CTC loss for speech recognition, but it's also possible for other kind of usage.
* In speech recognition case, CTC wraps the original sequence with blank every letter other one, e.g. C-A-T will be (blank)-C-(blank)-A-(blank)-T-(blank), and the network output dictionary += {(blank)}
* Operation B is defined to *"cut"* the network output to the target by removing the blank and the duplicated, e.g. B("A(blank)BBBC")=B("(blank)AAAAB(blank)CC")="ABC"
* The likelihood of the network output to become the target is the sum of all the situation. And the training is via maximizing likelihood.

![likelihood](http://img.blog.csdn.net/20150917194427949?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
* There will be so many cases to get the sum, thus a forward-backward algorithm like in HMM is designed to compute it. Besides, as can be shown in the paper, the likelihood is also **differentiable**.

![forward-backward](http://wantee.github.io/assets/images/posts/CTC-alpha-beta.png)
* CTC loss, as a general loss for temporal classification, can be in fact used in many kinds of network, though the dynamic programming approach is design for long sequences.
* Seq2Seq, CTC and house number recognizing (Goodfellow 2012) can be compared, for non fixed-length target output. 
