from src.mirror_bert import MirrorBERT

model_name = "cambridgeltl/mirror-roberta-base-sentence-drophead"
mirror_bert = MirrorBERT()
mirror_bert.load_model(path=model_name, use_cuda=True)

embeddings = mirror_bert.get_embeddings([
    'I transform pre-trained language models into universal text encoders.',
], agg_mode="tokens")
print (embeddings.shape)
examples = {'tokens': [['I', 'transform', 'pre-trained', 'language', 'models', 'into', 'universal', 'text', 'encoders', '.']], 
    'tags': [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']] }
label_to_id = {'O': 1}
label_all_tokens = False
tokenized_inputs = mirror_bert.tokenize_and_align_labels(examples=examples, label_to_id=label_to_id, 
    label_all_tokens=label_all_tokens, b_to_i_label=None)
print('tokenized_inputs: ')
print(tokenized_inputs)
print('done')