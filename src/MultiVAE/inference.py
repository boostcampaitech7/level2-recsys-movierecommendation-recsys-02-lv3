import torch
import numpy as np


def multivae_predict(args, model, data, k = 10):
    """
    Multi-VAE 모델을 사용하여 각 사용자별 Top-K 추천 아이템을 추출하는 함수
    
    Args:
        args: ArgumentParser에서 정의된 학습 및 환경 파라미터
        model: 학습된 Multi-VAE 모델
        data: 사용자-아이템 상호작용 데이터
        k: 추천할 아이템의 개수 (default: 10)

    Returns:
        top_items: 사용자별 Top-K 추천 아이템 리스트 [(user_id, item_id), ...]
    """
    
    model.eval()
    top_items = []
    
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data.toarray()).to(args.device)
        predicts = model(data_tensor)

        predicts = predicts[0]
        predicts[data.nonzero()] = -np.inf
        
        _  , top_ids = torch.topk(predicts.to(args.device).float(), k, dim=1)
        for user_id, item_ids in enumerate(top_ids):
            for item_id in item_ids:
                top_items.append((user_id, item_id.item()))

    return  predicts, top_items
