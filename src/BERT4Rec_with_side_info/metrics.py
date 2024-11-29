from tqdm import tqdm

def recall_at_k(num_user, transform_predict_df, tests_df):
    recalls = []
    for u in tqdm(range(num_user)):
        p = transform_predict_df[u]
        t = tests_df[u]

        intersection = list(set(p) & set(t))
        score = len(intersection) / 10
        
        recalls.append(score)
    print(f"recall@10: {sum(recalls) / len(recalls)}")