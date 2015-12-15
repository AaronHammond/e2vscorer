e2vscorer
=========

1. Set network topology parameters in net_config.py
2. Set structural topologies in prompt2vec_abs.py for the Absolute Prediction method and prompt2vec_class.py for the Categorical Prediction method
3. Train against the network configuration by running train_abs.py for Absolute Prediction, train_class.py for Categorical Prediction, and train_baseline.py for the Most-Frequent Baseline
	- data will be split automatically into training/testing data, and our performance metrics will be printed after the network trains for however many iterations are set in net_config.py

Note: use of virtualenv is _seriously_ recommended to mitigate problems with competing versions of numpy/gensim/scikit/keras/etc.

Note: the TensorFlow backend was used for Keras. No guarantees on whether all models will work as expected with the theanos backend

Note: if gensim's training of doc2vec is taking a painful amount of time (shouldn't be more than 30s or so on reasonable hardware), you should probably uninstall gensim, install cython and a C compiler, then reinstall gensim (https://radimrehurek.com/gensim/models/doc2vec.html#blog)

