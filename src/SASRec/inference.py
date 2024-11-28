import torch
import pandas as pd
import numpy as np

def predict(model, dataset, config):
    model.eval()
    predict_list = []
    
    for user in range(dataset.num_user):
        seq = dataset.user_train[user][-config['model']['max_len']:]
        
        item_indices = list(range(1, dataset.num_item + 1))
        
        with torch.no_grad():
            predictions = -model.predict(np.array([seq]), np.array(item_indices))
            predictions = predictions[0].cpu().numpy()
            
        partition_idx = np.argpartition(predictions, -10)[-10:]
        top_items = partition_idx[np.argsort(predictions[partition_idx])]
        
        user_id = dataset.idx2user[user]
        for item_idx in top_items:
            predict_list.append([user_id, dataset.idx2item[item_indices[item_idx]]])

    submission_df = pd.DataFrame(data=predict_list, columns=["user", "item"])
    submission_df = submission_df.sort_values("user")
    submission_df.to_csv('submission.csv', index=False)
    return submission_df