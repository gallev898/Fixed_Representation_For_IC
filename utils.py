import sys

from models.fixed_models_no_attention import DecoderWithoutAttention, Encoder

sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
# sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')


import json
import os

import torch
import torchvision.transforms as transforms


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
data_name = 'coco_5_cap_per_img_5_min_word_freq'
model_filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
output_folder_path = os.path.join(os.path.expanduser('~'), 'PycharmProjects/image_captioning/output_folder')
word_map_file_name = 'WORDMAP_' + data_name + '.json'
word_map_file = os.path.join(output_folder_path, word_map_file_name)
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
emb_dim = 512  # dimension of word embeddings
dropout = 0.5

def get_model_path_and_save_dir(args):
    if args.run_local:
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        trained_models_path = os.path.join(desktop_path, 'trained_models')
        model_path = os.path.join(trained_models_path, os.path.join(args.model, model_filename))
        save_dir = 'inference_data'
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, model_filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/inference_data".format(args.model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    if args.run_local:
        save_dir = os.path.join(save_dir, args.model)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print('created dir: {}'.format(save_dir))

    print('model path: {}'.format(model_path))
    print('save dir: {}'.format(save_dir))
    return model_path, save_dir



def get_models(model_path, device, vocab_size):
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    decoder = DecoderWithoutAttention(attention_dim=attention_dim,
                                      embed_dim=emb_dim,
                                      decoder_dim=decoder_dim,
                                      vocab_size=vocab_size,
                                      device=device,
                                      dropout=dropout)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)
    decoder.eval()

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder = encoder.to(device)
    encoder.eval()

    representations= checkpoint['representations']

    return encoder, decoder, representations


def get_word_map(args_):
    if args_.run_local:
        file = os.path.join(output_folder_path, word_map_file_name)
    else:
        server_path = '/yoav_stg/gshalev/image_captioning/output_folder'
        file = os.path.join(server_path, word_map_file_name)

    print('Loading word map from: {}'.format(file))
    with open(file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    return word_map, rev_word_map
# utils.py
