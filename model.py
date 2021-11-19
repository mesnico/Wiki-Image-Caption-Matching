import torch
from torch import nn
from transformers import AutoModel
import clip
from torch.nn import functional as F
from loss import ContrastiveLoss

class TextExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_model = config['text-model']['model-name']
        self.finetune = config['text-model']['finetune']
        self.text_model = AutoModel.from_pretrained(text_model)

    def forward(self, ids, mask):
        with torch.set_grad_enabled(self.finetune):
            out = self.text_model(input_ids=ids, attention_mask=mask, output_hidden_states=False)[1]
        #text_embeddings = self.text_fc(out)
        return out

class ImageExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.finetune = config['image-model']['finetune']
        model_name = config['image-model']['model-name']
        self.clip_model, _ = clip.load(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, img):
        with torch.set_grad_enabled(self.finetune):
            feats = self.clip_model.encode_image(img)
        return feats


class MatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_model = ImageExtractorModel(config)
        self.txt_model = TextExtractorModel(config)
        common_space_dim = config['matching']['common-space-dim']
        img_feat_dim = config['image-model']['dim']
        txt_feat_dim = config['text-model']['dim']
        self.url_fc = nn.Sequential(
            nn.Linear(txt_feat_dim, txt_feat_dim),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(txt_feat_dim),
            nn.ReLU(),
            nn.Linear(txt_feat_dim, txt_feat_dim),
            nn.BatchNorm1d(txt_feat_dim)
        )
        self.caption_fc = nn.Sequential(
            nn.Linear(txt_feat_dim, txt_feat_dim),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(txt_feat_dim),
            nn.ReLU(),
            nn.Linear(txt_feat_dim, common_space_dim),
            nn.BatchNorm1d(common_space_dim)
        )
        self.image_fc = nn.Sequential(
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(img_feat_dim),
            nn.ReLU(),
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.BatchNorm1d(img_feat_dim)
        )

        self.process_after_concat = nn.Sequential(
            nn.Linear(img_feat_dim + txt_feat_dim, img_feat_dim + txt_feat_dim),
            nn.ReLU(),
            nn.Linear(img_feat_dim + txt_feat_dim, common_space_dim)
        )

        contrastive_margin = config['training']['margin']
        max_violation = config['training']['max-violation']
        self.matching_loss = ContrastiveLoss(margin=contrastive_margin, max_violation=max_violation)

    def compute_embeddings(self, img, url, url_mask, caption, caption_mask):
        if torch.cuda.is_available():
            img = img.cuda()
            url = url.cuda()
            url_mask = url_mask.cuda()
            caption = caption.cuda()
            caption_mask = caption_mask.cuda()

        img_feats = self.img_model(img).float()
        img_feats = self.image_fc(img_feats)
        url_feats = self.txt_model(url, url_mask)
        url_feats = self.url_fc(url_feats)
        caption_feats = self.txt_model(caption, caption_mask)
        caption_feats = self.caption_fc(caption_feats)

        # concatenate img and url features
        query_feats = torch.cat([img_feats, url_feats], dim=1)
        query_feats = self.process_after_concat(query_feats)

        # L2 normalize output features
        query_feats = F.normalize(query_feats, p=2, dim=1)
        caption_feats = F.normalize(caption_feats, p=2, dim=1)

        return query_feats, caption_feats

    def compute_loss(self, query_feats, caption_feats):
        loss = self.matching_loss(query_feats, caption_feats)
        return loss

    def forward(self, img, url, url_mask, caption, caption_mask):
        # forward the embeddings
        query_feats, caption_feats = self.compute_embeddings(img, url, url_mask, caption, caption_mask)

        # compute loss
        loss = self.compute_loss(query_feats, caption_feats)
        return loss


