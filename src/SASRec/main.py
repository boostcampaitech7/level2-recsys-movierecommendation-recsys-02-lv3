from omegaconf import OmegaConf
import torch
from train import train, evaluate
from inference import predict

def main():
    config = OmegaConf.load('config.yaml')
    
    print("----------------Training----------------")
    model, dataset = train(config)
    
    print("----------------Evaluating----------------")
    metrics = evaluate(model, dataset, config)
    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"HIT@10: {metrics['HIT@10']:.4f}")
    
    print("----------------Predictions----------------")
    submission_df = predict(model, dataset, config)
    print("Saved predictions")

if __name__ == "__main__":
    main() 