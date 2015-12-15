e2vscorer
=========

1. Set network topology parameters in net_config.py
2. Set structural topologies in prompt2vec_abs.py for the Absolute Prediction method and prompt2vec_class.py for the Categorical Prediction method
3. Train against the network configuration by running train_abs.py for Absolute Prediction, train_class.py for Categorical Prediction, and train_baseline.py for the Most-Frequent Baseline
	- data will be split automatically into training/testing data, and our performance metrics will be printed after the network trains for however many iterations are set in net_config.py

Note: use of virtualenv is _seriously_ recommended to mitigate problems with competing versions of numpy/gensim/scikit/keras/etc.

Note: the TensorFlow backend was used for Keras. No guarantees on whether all models will work as expected with the theanos backend

Note: if gensim's training of doc2vec is taking a painful amount of time (shouldn't be more than 30s or so on reasonable hardware), you should probably uninstall gensim, install cython and a C compiler, then reinstall gensim (https://radimrehurek.com/gensim/models/doc2vec.html#blog)

Abstract:
While automatic essay grading remains a topic of interest in NLP, methods which seek to evaluate argument strength, instead of more formal dimensions, remain rare; furthermore, those described in the literature primarily rely on feature-based classifiers, whose heuristic rules can lose predictive value when applied across different essay sets. We propose a supervised, neural network-based approach, based on word2vec and one of its extensions, doc2vec, to construct a classifier over trained essay and prompt embeddings; this classifier provides a means for scoring essays on the basis of the quality of their argument in answering the relevant prompt without a priori feature extraction or annotation. Our methods significantly outperform a frequency-based baseline, achieving upwards of 75-80% accuracy on unseen student essays drawn from two distinct prompt-response sets
