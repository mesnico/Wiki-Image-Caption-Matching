import time
import numpy as np
import torch
import numpy
import tqdm
from collections import OrderedDict


def encode_data(model, data_loader, log_step=100000000, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """

    # switch to evaluate mode
    model.eval()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # ids_pointer = 0
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description('Encoding validation data')
    for i, data in enumerate(pbar):
        # bs = len(data[0])
        # ids = list(range(ids_pointer, ids_pointer + bs))
        # ids_pointer += bs

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb = model.compute_embeddings(*data)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = img_emb.cpu()
                cap_embs = cap_emb.cpu()
            else:
                img_embs = torch.cat([img_embs, img_emb.cpu()], dim=0)
                cap_embs = torch.cat([cap_embs, cap_emb.cpu()], dim=0)

            # preserve the embeddings by copying from gpu and converting to numpy
            # img_embs[ids, :] = img_emb.cpu()
            # cap_embs[ids, :] = cap_emb.cpu()

            # measure accuracy and record loss
            # model.forward_loss(None, None, img_emb, cap_emb, img_length, cap_length)

        if (i + 1) % log_step == 0:
            logging('Test: [{0}/{1}]'
                    .format(
                        i, len(data_loader)))

    return img_embs, cap_embs


def compute_recall(queries, captions):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    npts = queries.shape[0]

    ranks = numpy.zeros(npts)
    pbar = tqdm.trange(npts)
    pbar.set_description('Validation')
    for index in pbar:

        # Get query image-url
        query = queries[index].unsqueeze(0)
        # Compute scores
        d = torch.mm(query, captions.T)     # 1 x npts
        d = d.cpu().numpy().flatten()

        ordered = numpy.argsort(d)[::-1]
        ranks[index] = numpy.where(ordered == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return {'r1':r1, 'r5': r5, 'r10':r10, 'medr':medr, 'meanr':meanr}