from src.mirror_bert import MirrorBERT
from collections import defaultdict

def test_case():
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

def main():
    # Create examples
    with open('../automl/data/atis/train/seq.in', 'r') as text_reader:
        with open('../automl/data/atis/train/seq.out') as label_reader:
            all_words = []
            all_tags = []
            label_to_id = defaultdict(lambda: len(label_to_id) + 1)
            for text, label in zip(text_reader, label_reader):
                words = text.strip().split()
                tags = label.strip().split()
                all_words.append(words)
                all_tags.append(tags)
                assert len(words) == len(tags)
            examples = {'tokens': all_words, 'tags': all_tags}
            mirror_bert = MirrorBERT()
            model_name = "cambridgeltl/mirror-roberta-base-sentence-drophead"
            mirror_bert.load_model(path=model_name, use_cuda=True)
            label_all_tokens = False
            tokenized_inputs = mirror_bert.tokenize_and_align_labels(examples=examples, label_to_id=label_to_id, 
                label_all_tokens=label_all_tokens, b_to_i_label=None)
            # Now I need to compute the features for each token
            embeddings = mirror_bert.get_embeddings(examples['tokens'], agg_mode="tokens")
            print (embeddings.shape)
            print('we have all the features, now generate the vw data!')

if __name__ == '__main__':
    main()