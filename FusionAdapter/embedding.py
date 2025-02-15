import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.es = parameter['embed_dim']

        num_ent = len(self.ent2id)
        self.embedding = nn.Embedding(num_ent, self.es)

        self.image = nn.Embedding(11757, 4096)
        self.text = nn.Embedding(11757, 1000)
        self.text2emb = dataset['text2emb']
        self.visual2emb = dataset['visual2emb']
        self.struc2emb = dataset['struc2emb']


        self.image.weight.data.copy_(torch.from_numpy(self.visual2emb))
        self.text.weight.data.copy_(torch.from_numpy(self.text2emb))
        self.embedding.weight.data.copy_(torch.from_numpy(self.struc2emb))



    def forward(self, triples,adapter_image, adapter_text):
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)

        # Compute individual modality embeddings
        structural_embedding = self.embedding(idx)  # Structure modality
        image_modality = adapter_image(self.image(idx))  # Image modality
        text_modality = adapter_text(self.text(idx))  # Text modality

        # Combined embedding
        combined_embedding = structural_embedding + image_modality + text_modality

        return combined_embedding, structural_embedding, image_modality, text_modality




