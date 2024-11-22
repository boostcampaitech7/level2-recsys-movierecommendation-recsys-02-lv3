import torch
import numpy as np


def ease_predict(args, model, data, k):
    result = model.forward(data[:,:])
    result[data.nonzero()] = -np.inf
    
    top_items = []
    _  , top_ids = torch.topk(torch.from_numpy(result).to(args.device), k, dim=1)
    for user_id, item_ids in enumerate(top_ids):
            for item_id in item_ids:
                top_items.append((user_id, item_id.item()))

    return top_items