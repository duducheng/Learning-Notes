# [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078)

Notes:
My first reading on visualizing RNN. This work is based on their Char-RNN, trying to understand 

1. internal mechanisms of LSTM: 
    * on LSTM cells' outputs: some cells well segment the sequence;    
    * gates' activation: use "saturation" to tell the behavioir of the gates. Most gates' behavior much differ from the feed forward network: LSTM's gates can be almost always closed to 0 (forget) or 1 (memory). Except for the first layers.

![LSTM_cell](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/40be3888daa5c2e5af4d36ae22f690bcc8caf600/4-Figure2-1.png)

![LSTM_gate](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/40be3888daa5c2e5af4d36ae22f690bcc8caf600/4-Figure3-1.png)

2. long-range interaction: compared to n-NN and n-grams (20-grams), LSTM outperforms in accuracy, cross-entropy and model size. When doing error analysis, the disjoint error cases of LSTM is much less.

3. break-down failure cases: too specific to NLP, but note that they do this by selecting the error cases manually. **They found simply scaling up the model can reduce the local (n-gram) error, but leave other untouched.**