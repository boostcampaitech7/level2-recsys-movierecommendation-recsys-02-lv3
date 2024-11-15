import torch
import numpy as np


def predict(args, model, data_tr):
    model.eval()
    
    def _naive_sparse_to_tensor(data):
        return torch.FloatTensor(data.toarray())    
    
    top_items = []
    
    with torch.no_grad():
        data_tensor = _naive_sparse_to_tensor(data_tr).to(args.device)
        predicts = model(data_tensor)

        predicts = predicts[0].cpu().numpy()    

        predicts[data_tr.nonzero()] = -np.inf
        
        top_scores, top_ids = torch.topk(torch.from_numpy(predicts).float().to(args.device), k=10, dim=1)
        for user_id, item_ids in enumerate(top_ids):
            for item_id in item_ids:
                top_items.append((user_id, item_id.item()))


    return top_items