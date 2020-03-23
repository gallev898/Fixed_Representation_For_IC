import os
import json
import torch
import torch.nn.functional as F
import argparse

# section: args
parser = argparse.ArgumentParser(description='embeddings_sematic_analysis')
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--top_k', type=int)
parser.add_argument('--model', type=str)
args = parser.parse_args()

# section: initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
model_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(args.model, model_name)
data_folder = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'

# section: load representation
# NOTICE: TEMP
representations = torch.load('/Users/gallevshalev/Desktop/{}_representations'.format(args.model))['representations']
# subsec: normalize representation
representations = F.normalize(representations, dim=0)
assert round(torch.norm(representations[:, 9489]).item()) == 1

# section: word map
if not args.run_local:
    data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
else:
    data_f = data_folder

word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
print('word_map_file: {}'.format(word_map_file))

print('loading word map from path: {}'.format(word_map_file))
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
print('load word map COMPLETED')

rev_word_map = {v: k for k, v in word_map.items()}

# section: cosine
original_words = 'sitting', 'dog', 'table', 'car', 'apple'
cosine_word = []
for ow in original_words:
    cosine_word.append(representations[:, word_map[ow]])

for ow, word in zip(original_words, cosine_word):
    cosine = torch.matmul(word, representations)
    top_k = torch.topk(cosine, args.top_k)
    indices = top_k.indices
    values = top_k.values
    similarities = []
    for ind, val in zip(indices, values):
        similarities.append('{}_{:.2f}'.format(rev_word_map[ind.item()], val))
    print('{}   -{}'.format(ow, ', '.join(similarities)))

