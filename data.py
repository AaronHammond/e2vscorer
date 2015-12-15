import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data_file_path = 'data/training_set_rel3.tsv'

import net_config as config
import linecache
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pickle

def compute_score_histogram(score_field, filterer = None):
	score_counts = {}

	for essay in get_essays(filterer = filterer):
		new_score = essay[score_field]
		score_counts[new_score] = (score_counts.get(new_score, 0) + 1)

	norm = float(sum(score_counts.values()))
	hist = {}

	for k in score_counts:
		hist[k] = score_counts[k] / norm

	return hist


def get_essays(filterer = None, post_processor = None, limit = float('inf')):
	returned = 0
	parsed = 0
	while parsed < config.essay_count and returned < limit:
		essay_index = _essay_shuffle[parsed]
		line = linecache.getline(data_file_path, essay_index + 1) # header
		essay = construct_essay(line)
		parsed+=1

		if filterer is None or filterer(essay):
			if post_processor is not None:
				yield post_processor(essay)
			else:
				yield essay
			returned+=1

def construct_essay(line):
	data = re.split("\t", line)

	if not data[0].isdigit() or not data[1].isdigit():
		raise ValueError()

	essay = {}

	essay["essay_id"] = int(data[0])
	essay["prompt_id"] = int(data[1])

	essay["rater_1_domain_1"] = if_int(data[3])
	essay["rater_2_domain_1"] = if_int(data[4])
	essay["rater_3_domain_1"] = if_int(data[5])
	
	essay["domain_1_resolved"] = if_int(data[6])

	essay["rater_1_domain_2"] = if_int(data[7])
	essay["rater_2_domain_2"] = if_int(data[8])

	essay["domain_2_resolved"] = if_int(data[9])

	return essay

def if_int(datum):
	if datum.isdigit():
		return int(datum)
	else:
		return None


_e2v = None
def initE2V():
	global _e2v
	global _essay_shuffle
	config.essay_count = (sum(1 for line in open(data_file_path) if line.rstrip()) - 1)

	essay_index = 1
	docs = []
	while essay_index <= config.essay_count:
		line = linecache.getline(data_file_path, essay_index + 1) # header
		parts = re.split("\t", line)

		essay_id = int(parts[0])
		raw_text = parts[2]
		
		words = [w.lower() for w in re.split("[^\w'@]+", re.sub("(\s+)?[\.\?](\s+)?", " @stop ", raw_text))]
		docs.append(TaggedDocument(words, [essay_id]))
		essay_index+=1

	_e2v = Doc2Vec(docs, size=config.essay_embedding_size, window=20, min_count=5, workers=4, negative=5)
	_e2v.init_sims(replace=True)

	print "randomly ordering essays..."
	# _essay_shuffle = np.arange(1, config.essay_count+1)
	# np.random.shuffle(_essay_shuffle)
	# pickle.dump(_essay_shuffle, open('../data/shuffle', 'w'))
	_essay_shuffle = pickle.load(open('data/shuffle', 'r'))
	print "done randomly ordering essays"

def essay2vec(essay_id):
	return _e2v.docvecs[essay_id]
