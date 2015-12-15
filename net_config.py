essay_embedding_size = 150
prompt_embedding_size = 600
essay_hidden_layer_size = 128
prompt_hidden_layer_size = 128
hidden_layer_size = 256

targeted_prompts = [1, 2]
targeted_field = "rater_1_domain_1"
validation_fields = ["rater_1_domain_1", "rater_2_domain_1"]
score_range = (1, 6)

prompt_count = 8
essay_count = None # populated during data extraction

p2v_training_iterations = 200
testing_slice_size = 0.2

per_prompt_limit = 2000
batch_size = 8

optimizer = 'adam'

def summary():
	print "Targeted field", targeted_field
	print "Validation fields", validation_fields
	print "Targeted prompts", targeted_prompts
	print "Prompt Embedding Size", prompt_embedding_size
	print "Essay Embedding Size", essay_embedding_size
	print "Essay Hidden Layer Size", essay_hidden_layer_size
	print "Prompt Hidden Layer Size", prompt_hidden_layer_size
	print "Hidden Layer Size", hidden_layer_size
	print "Optimizer", optimizer
	print "Batch Size", batch_size

