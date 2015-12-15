import net_config as config
import data as d
d.initE2V()
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
assignment = sorted([(score_histogram[s], s) for s in score_histogram.keys()], reverse=True)[0][1]
print "Score Histogram", score_histogram
print "Will Assign", assignment

def evaluate(essays_it):
	total_ct = 0
	correct_ct = 0
	acceptable_ct = 0
	diff_g_1 = 0
	for essay in essays_it:
		actual = essay[config.targeted_field]
		predicted = assignment
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
print "Method: Most-Frequent Baseline"
print "------------"
test_all()
