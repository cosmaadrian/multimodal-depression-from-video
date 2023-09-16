import torch
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import torch
import wandb
from torch.utils.data import Dataset
import yaml
import os

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pprint

import callbacks
from trainer import NotALightningTrainer
from loggers import WandbLogger

from utils import load_args, load_model

import nomenclature
import torch

from sentence_transformers import SentenceTransformer
from transformers import ViTFeatureExtractor, ViTModel, CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaTokenizer, BertModel, BertTokenizer, AutoModel, AutoTokenizer

from datasets import TwitterDataset
from datasets import TwitterSubmissionDataset

from captum.attr import IntegratedGradients

KIND = 'positive'

parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('--config_file', type = str, required = False, default = './configs/combos/clip_emoberta.yaml')

parser.add_argument('--name', type = str, default = 'fold-1-twitter-ws-128-clip-emoberta-time2vec')
parser.add_argument('--group', type = str, default = 'final-final-time2vec')

parser.add_argument('--window_size', type = int, default = 128)
parser.add_argument('--position_embeddings', type = str, default = 'time2vec')
parser.add_argument('--image_embeddings_type', type = str, default = None)
parser.add_argument('--text_embeddings_type', type = str, default = None)
parser.add_argument('--fold', type = int, default = 1)

parser.add_argument('--batch_size', type = int, default = 16)

args = parser.parse_args()
args, cfg = load_args(args)

class PatchedDataset(Dataset):
    def __init__(self, args, time_dataset, image_encoder, text_encoder, label = 'positive', users = None):
        self.args = args
        self.label = label
        self.time_dataset = time_dataset
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.original_dataset = TwitterSubmissionDataset(args, label = label, users = users)

    def get_batch(self, idx):
        output = self.original_dataset[idx]

        output['date'] = np.array(output['date'][-self.args.window_size:])
        output['texts'] = output['texts'][-self.args.window_size:]
        output['images_paths'] = output['images_paths'][-self.args.window_size:]
        output['images'] = output['images'][-self.args.window_size:]
        output['ids'] = output['ids'][-self.args.window_size:]


        # output['date'] = np.array(output['date'])
        # output['texts'] = output['texts']
        # output['images_paths'] = output['images_paths']
        # output['images'] = output['images']
        # output['ids'] = output['ids']

        print(":: Extracting image embeddings")
        image_embeddings = extract_image_embedding(output['images'], model = self.image_encoder[0], input_representation = self.image_encoder[1])

        print(":: Extracting text embeddings")
        text_embeddings = extract_text_embedding_sent(output['texts'], model = self.text_encoder)

        print(":: Loading multimodal batch")
        batch = self.time_dataset.load_multimodal(
                image_embeddings = image_embeddings,
                text_embeddings = text_embeddings,
                label = 1 if self.label == 'positive' else 0,
                dates = output['date'],
                user_name = output['author'],
            )

        batch['texts'] = output['texts']
        batch['images_paths'] = output['images_paths']

        return batch

class ModelAdaptor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text_embeddings, text_mask, image_embeddings, image_mask, time):
        output = self.model({
                'text_embeddings': text_embeddings,
                'text_mask': text_mask,
                'image_embeddings': image_embeddings,
                'image_mask': image_mask,
                'time': time
            }
        )

        return output['probas']

def extract_text_embedding_sent(user_texts, model):
    BATCH_SIZE = 128
    texts = []
    for post in tqdm(user_texts):
        encoded_texts = model.encode(post, batch_size = args.batch_size, convert_to_numpy = True)
        texts.append(encoded_texts)
    return np.array(texts)

def extract_image_embedding(user_images, model, input_representation):
    embeddings_list = []
    for user_image in tqdm(user_images):
        if user_image is np.nan or user_image.shape == (1, 1, 3):
            embedding = np.nan
        else:
            with torch.no_grad():
                user_image = user_image.transpose(2, 0, 1)
                inputs = input_representation(images = user_image, return_tensors="pt")
                out = model(pixel_values = inputs['pixel_values'].cuda())
                # last_hidden = out.last_hidden_state.mean(dim = 1)
                last_hidden = out.pooler_output
                embedding = last_hidden.detach().cpu().numpy()

        embeddings_list.append(embedding)

    return embeddings_list

embedders = {
    'dino': {
        'model_name': 'facebook/dino-vitb16',
        'model_class': ViTModel,
        'input_representation': ViTFeatureExtractor
    },
    'clip': {
        'model_name': 'openai/clip-vit-base-patch16',
        'model_class': CLIPVisionModel,
        'input_representation': CLIPProcessor
    },
    'roberta': {
        'model_name': 'sentence-transformers/stsb-roberta-base',
        'model_class': RobertaModel,
        'input_representation': RobertaTokenizer
    },
    'emoberta':{
        'model_name': 'tae898/emoberta-base',
        'model_class': AutoModel,
        'input_representation': AutoTokenizer
    },
}

image_input_representation = embedders[args.image_embeddings_type]['input_representation'].from_pretrained(embedders[args.image_embeddings_type]['model_name'])
image_model = embedders[args.image_embeddings_type]['model_class'].from_pretrained(embedders[args.image_embeddings_type]['model_name']).cuda().eval()

text_encoder = SentenceTransformer(embedders[args.text_embeddings_type]['model_name']).cuda().eval()

model = nomenclature.MODELS[args.model](args)

state_dict = load_model(args)
state_dict = {
    key.replace('module.', ''): value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict)

model_adaptor = ModelAdaptor(model = model)
model_adaptor.eval()
model_adaptor.train(False)
# model_adaptor.cuda()

ig = IntegratedGradients(model_adaptor)

twitter_dataset = TwitterDataset(args, kind = 'test')
users = twitter_dataset.negative_users if KIND == 'negative' else twitter_dataset.positive_users

patched_dataset = PatchedDataset(
    args,
    time_dataset = twitter_dataset,
    image_encoder = (image_model, image_input_representation),
    text_encoder = text_encoder,
    label = KIND,
    users = users,
)

for j in range(35, len(patched_dataset.original_dataset)):
    text_encoder.cuda()
    image_model.cuda()

    data = patched_dataset.get_batch(j)

    text_encoder.cpu()
    image_model.cpu()

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = torch.from_numpy(value).unsqueeze(0)#.cuda()

    inputs = (data['text_embeddings'], data['text_mask'], data['image_embeddings'], data['image_mask'], data['time'])
    pred = model_adaptor(*inputs)

    # if KIND == 'negative' and round(pred[0][0].item()) == 0:
    #     continue

    # if KIND == 'positive' and round(pred[0][0].item()) == 1:
    #     continue

    print("INCORECT PREDICTION FOR USER:", j)

    baselines = (torch.zeros_like(inputs[0]), torch.zeros_like(inputs[1]), torch.zeros_like(inputs[2]), torch.zeros_like(inputs[3]), torch.zeros_like(inputs[4]))

    print(":: Extracting attributions")
    attributions = ig.attribute(
        inputs,
        baselines,
        return_convergence_delta = False
    )

    text_attributions = attributions[0].sum(axis = -1).squeeze(0)
    text_attributions = text_attributions / torch.norm(text_attributions)
    text_attributions = text_attributions.detach().cpu().numpy()

    image_attributions = attributions[2].sum(axis = -1).squeeze(0)
    image_attributions = image_attributions / torch.norm(image_attributions)
    image_attributions = image_attributions.detach().cpu().numpy()

    date_attribution = attributions[4].sum(axis = -1).squeeze(0)
    date_attribution = date_attribution / torch.norm(date_attribution)
    date_attribution = date_attribution.detach().cpu().numpy()

    print(":::::: TEXT ATTRIBUTION")
    for i, (text, attribution) in enumerate(sorted(zip(data['texts'], text_attributions[:len(data['texts'])]), key = lambda x: x[1])):
        print(i, attribution, text[:])

    print(":::::: IMAGE ATTRIBUTION")
    for i, (image_path, attribution, _) in enumerate(sorted(zip(data['images_paths'], image_attributions[:len(data['images_paths'])], text_attributions[:len(data['texts'])]), key = lambda x: x[-1])):
        print(i, attribution, image_path)

    print("=== Predition: ", pred)
