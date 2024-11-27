import pandas as pd
import numpy as np
import torch

def bert4rec_predict(model, users, user_train, num_user, num_item, max_len, idx2user, idx2item, output_path="submission.csv", predict_save_path="predictions.npy", device="cuda"):

    """
    모델을 사용해 추천 결과를 생성하고 제출 파일로 저장하는 함수입니다

    Args:
        model: 학습된 모델
        users: 전체 사용자 리스트
        user_train: 사용자별 시퀀스 데이터
        num_user: 총 사용자 수
        num_item: 총 아이템 수
        max_len: 최대 시퀀스 길이
        idx2user: 인덱스를 원래 사용자 ID로 매핑하는 맵
        idx2item: 인덱스를 원래 아이템 ID로 매핑하는 맵
        output_path: 저장 경로
        device: 연산에 사용할 디바이스
    
    Returns:
        None: 결과를 CSV 파일로 저장하며, 반환값은 없습니다.
        저장된 파일: output_path 위치에 user-item 추천 결과가 저장됩니다.
    """
    model.eval()
    predicted_user = []
    predicted_item = []
    scores_matrix = np.zeros((num_user, num_item))  # 예측 점수를 저장할 numpy 배열

    for u in users:
        seq = (user_train[u] + [num_item + 1])[-max_len:]  # max_len 크기로 패딩
        rated = set(user_train[u])

        # 모든 아이템을 대상으로 추천 점수 계산
        item_idx = list(range(1, num_item + 1))  # 전체 아이템

        with torch.no_grad():
            predictions = -model(torch.tensor([seq], dtype=torch.long).to(device))
            predictions = predictions[0, -1, item_idx]  # 마지막 타임스텝 예측 결과
            scores_matrix[u, :] = predictions.cpu().numpy()  # 점수 저장

            top_10_indices = predictions.argsort()[:10]  # 상위 10개 추천
            recommended_items = [item_idx[i] for i in top_10_indices]  # 추천된 아이템

        # 추천 결과 저장
        predicted_user.extend([u] * len(recommended_items))  # 유저 ID 추가
        predicted_item.extend(recommended_items)  # 추천된 아이템 추가

    # numpy 배열 저장
    np.save(predict_save_path, scores_matrix)
    print(f"Predictions saved as numpy array at {predict_save_path}")

    # 추천 결과 저장
    submission_df = pd.DataFrame({'user': predicted_user, 'item': predicted_item}) # DataFrame 생성
    submission_df['user'] = submission_df['user'].map(idx2user) # 역매핑하여 원래 사용자 및 아이템 ID로 변환
    submission_df['item'] = submission_df['item'].map(idx2item)
    submission_df.to_csv(output_path, index=False) # 제출 파일 저장
    print(f"Submission saved to {output_path}!")
