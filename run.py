from src.mirror_bert import MirrorBERT
from collections import defaultdict
import json

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
    split = 'test'
    dataset = 'CustomerData/Novaratis'
    with open('../automl/data/'+dataset+'/'+split+'/label_to_int.txt', 'r') as reader:
        label_to_id = json.load(reader)
    with open('../automl/data/'+dataset+'/'+split+'/seq.in', 'r') as text_reader:
        with open('../automl/data/'+dataset+'/'+split+'/seq.out', 'r') as label_reader:
            all_words = []
            all_tags = []
            for text, label in zip(text_reader, label_reader):
                words = text.strip().split()
                tags = label.strip().split()
                all_words.append(words)
                all_tags.append(tags)
                assert len(words) == len(tags)
            examples = {'tokens': all_words, 'tags': all_tags}

            # TODO we want to also support tacl features here
            mirror_bert = MirrorBERT()
            model_name = 'cambridgeltl/mirror-roberta-base-sentence-drophead'
            mirror_bert.load_model(path=model_name, use_cuda=True)
            label_all_tokens = False

            tokenized_inputs = mirror_bert.tokenize_and_align_labels(examples=examples, label_to_id=label_to_id, 
                label_all_tokens=label_all_tokens, b_to_i_label=None)

#            label_list = label_to_id.keys()
#            # Map that sends B-Xxx label to its I-Xxx counterpart
#            b_to_i_label = []
#            for idx, label in enumerate(label_list):
#                if label.startswith("B-") and label.replace("B-", "I-") in label_list:
#                    b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
#                else:
#                    b_to_i_label.append(idx)

            # Now I need to compute the features for each token
            embeddings = mirror_bert.get_embeddings(examples['tokens'], agg_mode="tokens")
            print (embeddings.shape)
            print('we have all the features, now generate the vw data!')
            with open('../automl/data/'+dataset+'/'+split+'/'+dataset.split('/')[-1]+'_mirror.vw', 'w') as vw_writer:
                with open('../automl/data/'+dataset+'/'+split+'/'+dataset.split('/')[-1]+'_unigram.vw', 'w') as unigram_writer:
                    for inpt, mask, label, feature in zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], 
                        tokenized_inputs['labels'], embeddings):
                        inpt = inpt[mask==1]
                        label = label[mask==1]
                        feature = feature[mask==1]
                        assert len(inpt) == len(label)
                        assert len(label) == len(feature)
                        for l, f, i in zip(label, feature, inpt):
                            if l.item() != -100:
                                unigram_feature = ' ' + str(i.item()) + ' '
                                vw_features = ' '.join([str(i) + ':' + f"{x:.3f}" for i, x in enumerate(f)])
                                vw_writer.write(str(l.item()) + ' |w ' + vw_features + '\n')
                                unigram_writer.write(str(l.item()) + ' |w ' + unigram_feature + '\n')
                        vw_writer.write('\n')
                        unigram_writer.write('\n')
            print('done creating vw data')

if __name__ == '__main__':
    main()
