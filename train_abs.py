import net_config as config
import data as d
d.initE2V()
import prompt2vec_abs as p2v
import numpy as np

def prompt_filter(prompts):
	counts = {}
	
	def filterer(essay):
		if essay["prompt_id"] in prompts:
			counts[essay["prompt_id"]] = (counts.get(essay["prompt_id"], 0) + 1)
			return counts[essay["prompt_id"]] < config.per_prompt_limit
		return False

	return filterer

#
# BEGIN Training and Testing Sets Selection
#
training_set = {}
for essay in d.get_essays(filterer=prompt_filter(config.targeted_prompts)):
	if np.random.rand() > config.testing_slice_size:
		training_set[essay["essay_id"]] = True
	else:
		training_set[essay["essay_id"]] = False

def training_and_prompt_filter(prompts):
	global training_set

	pf = prompt_filter(prompts)
	def filterer(essay):
		return pf(essay) and training_set[essay["essay_id"]]

	return filterer

def testing_and_prompt_filter(prompts):
	global training_set

	pf = prompt_filter(prompts)
	def filterer(essay):
		return pf(essay) and not training_set[essay["essay_id"]]

	return filterer
#
# END Training and Testing Sets Selection
#

#
# BEGIN Score Ranges Computation
#
score_histogram = d.compute_score_histogram(config.targeted_field, filterer=training_and_prompt_filter(config.targeted_prompts))
def score_to_float(score):
	global score_histogram
	prob_mass = 0
	for i in range(config.score_range[0], score):
		prob_mass += score_histogram.get(i, 0)
	return prob_mass + (score_histogram.get(score, 0) / 2)

def float_to_score(flo):
	global score_histogram
	prob_mass = 0
	for i in range(1, config.score_range[1]):
		prob_mass += score_histogram.get(i, 0)

		if flo < prob_mass:
			return i
	return config.score_range[1]
#
# END Score Ranges Computation
#

#
# BEGIN p2v training
#
def p2v_iterations():
	training_count = sum(training_set.values())
	X_prompt = np.empty([training_count, 1])
	X_ess = np.empty([training_count, config.essay_embedding_size])
	y_score = np.empty([training_count, 1])
	for i, essay in enumerate(d.get_essays(filterer = training_and_prompt_filter(config.targeted_prompts))):
		X_prompt[i] = np.array([ essay["prompt_id"] ])
		X_ess[i] = np.array(d.essay2vec(essay["essay_id"]))
		y_score[i] = np.array([ score_to_float(essay[config.targeted_field]) ])

	testing_count = len(training_set.keys()) - training_count
	X_prompt_val = np.empty([testing_count, 1])
	X_ess_val = np.empty([testing_count, config.essay_embedding_size])
	y_score_val = np.empty([testing_count, 1])
	for i, essay in enumerate(d.get_essays(filterer = testing_and_prompt_filter(config.targeted_prompts))):
		X_prompt_val[i] = np.array([ essay["prompt_id"] ])
		X_ess_val[i] = np.array(d.essay2vec(essay["essay_id"]))
		y_score_val[i] = np.array([ score_to_float(essay[config.targeted_field]) ])

	p2v.out_score.fit([X_prompt, X_ess], y_score, verbose=1, batch_size=config.batch_size, nb_epoch=config.p2v_training_iterations, validation_data=([X_prompt_val, X_ess_val], y_score_val))
#
# END p2v training
#
#
# BEGIN p2v testing
#

def evaluate(essays_it):
	total_ct = 0
	correct_ct = 0
	acceptable_ct = 0
	diff_g_1 = 0
	for essay in essays_it:
		X_ess = np.array([ d.essay2vec(essay["essay_id"]) ])
		X_prompt = np.array([ [ essay["prompt_id"] ] ])
		prediction_raw = p2v.out_score.predict([ X_prompt, X_ess ], batch_size=1)[0]

		actual = essay[config.targeted_field]
		predicted = float_to_score(prediction_raw)
		total_ct+=1
		acceptable = [essay[f] for f in config.validation_fields]

		if actual == predicted:
			correct_ct+=1

		if predicted in acceptable:
			acceptable_ct+=1

		if abs(actual - predicted) > 1:
			diff_g_1+=1

	print "Absolute accuracy over set", (correct_ct / float(total_ct))
	print "Acceptable accuracy over set", (acceptable_ct / float(total_ct))
	print "Difference > 1 over set", (diff_g_1 / float(total_ct))

def test_all():
	print "Training Set"
	evaluate(d.get_essays(filterer=training_and_prompt_filter(config.targeted_prompts)))
	print "Testing Set"
	evaluate(d.get_essays(filterer=testing_and_prompt_filter(config.targeted_prompts)))
#
# END p2v testing
#
print "------------"
print "Method: Absolute Prediction"
config.summary()
print "------------"
p2v_iterations()
test_all()
