import time
import os
from tqdm import tqdm
from copy import deepcopy

from dataset import tensor_to_csr
import numpy as np
import torch
import torch.optim as optimizer_module

from utils.metrics import Recall_at_k_batch

def run(model, opts, train_data, n_epochs, batch_size, beta, gamma, dropout_rate, device='cuda'):
    model.train()
    for epoch in range(n_epochs):
        for idx, batch in enumerate(train_data):
            data = batch.to(device)

            for opt in opts:
                opt.zero_grad()

            _, loss = model(batch, beta=beta, gamma=gamma)
            loss.backward()

            for opt in opts:
                opt.step()


def train(args, model, train_dataset, valid_dataset, logger, setting):
    
    # Optimizer and Scheduler 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_encoder = getattr(optimizer_module, args.optimizer.type)(trainable_params,
                                                               **args.optimizer.args)    
    optimizer_decoder = getattr(optimizer_module, args.optimizer.type)(trainable_params,
                                                               **args.optimizer.args)    
    if args.not_alternating:
        opts = [optimizer_encoder, optimizer_decoder]

    # learning parameter 설정
    learning_kwargs = {
        'model': model,
        'n_epochs':3,
        'train_data': train_dataset,
        'batch_size': args.batch_size,
        'beta': args.beta,
        'gamma': args.gamma,
        'device': 'cuda'
    }

    # loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False)
    
    train_scores = []
    
    def implicit_slim(W, X, λ, α, thr):
        A = W.copy().astype(np.float16)
        
        D = 1 / (np.array(X.sum(0)) + λ)
        
        ind = (np.array(X.sum(axis=0)) < thr).squeeze()
        A[:, ind.nonzero()[0]] = 0
        
        M = (λ * A + A @ X.T @ X) * D * D
        
        AinvC = λ * M + M @ X.T @ X
        AinvCAt = AinvC @ A.T

        AC = AinvC - AinvCAt @ np.linalg.inv(np.eye(A.shape[0]) / α + AinvCAt) @ AinvC
        
        return α * W @ A.T @ AC


    for epoch in range(args.train.epochs):
        if args.implicitslim and epoch % args.step == args.step - 1:
            encoder_embs = model.encoder.fc1.weight.data
            decoder_embs = model.decoder.weight.data.T
            for embs in [encoder_embs, decoder_embs]:
                embs[:] = torch.Tensor(
                    implicit_slim(embs.detach().cpu().numpy(), train_dataset, args.lambd, args.alpha, args.threshold)
                ).to(args.device)
        
        if args.not_alternating:
            run(opts=[optimizer_encoder, optimizer_decoder], args=args, n_epochs=1, dropout_rate=0.5, **learning_kwargs)
        else:
            run(opts=[optimizer_encoder], args=args, n_epochs=args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
            model.update_prior()
            run(opts=[optimizer_decoder], args=args, n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)

        train_scores.append(
            evaluate(args, model, train_loader)
        )
        
        r10 = evaluate(args, model, valid_loader)
        
        if r10 > best_r10:
            best_r10 = r10
            os.makedirs(args.train.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt')
            

        print(f'epoch {epoch} | valid recall@10: {r10:.4f} | ' +
            f'best valid: {best_r10:.4f} | train recall@10: {train_scores[-1]:.4f}')

    return model

def evaluate(args, model, dataloader):
    model.eval()
    
    # Recall r10
        
    # DataLoader 생성
    global update_count
    r10_list = []

    with torch.no_grad():
        for _, data in enumerate(dataloader):            
            data_tr, data_te = data[0].squeeze(1).to(args.device), tensor_to_csr(data[1].squeeze(1).to(args.device))
            # 배치 크기와 형태 확인
            
            if args.other_params.args.total_anneal_steps > 0:
                anneal = min(args.other_params.args.anneal_cap,
                             1. * update_count / args.other_params.args.total_anneal_steps)
            else:
                anneal = args.other_params.args.anneal_cap

            # 모델 예측
            output = model(data_tr)

            # Exclude examples from training set
            recon_batch = output[0].cpu().numpy()
            data_tr = data_tr.cpu().numpy()

            # Training data에서 이미 평가에 포함된 항목 제외
            recon_batch[data_tr.nonzero()] = -np.inf

            # 평가 지표 계산
            r10 = Recall_at_k_batch(recon_batch, data_te, k=10)
            r10_list.append(r10)

    # 결과 평균 계산
    r10_list = np.concatenate(r10_list)

    return np.nanmean(r10_list)
    


def test(args, model, dataset, setting, checkpoint = None):
    
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        if args.train.save_best_model:
            model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt'
        else:
            # best가 아닐 경우 마지막 에폭으로 테스트하도록 함
            model_path = f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{args.train.epochs-1:02d}.pt'
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Run on test data
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.dataloader.batch_size, shuffle=False)

    test_loss, r10 = evaluate(args, model, test_loader)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} |  r10 {:4.2f} | '.format(test_loss, r10))
    print('=' * 89)
    
    return model