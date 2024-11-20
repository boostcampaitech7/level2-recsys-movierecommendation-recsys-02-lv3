import torch
import time
import numpy as np
import os
from src.loss.loss_fn import multivae_loss
from src.utils.metrics import Recall_at_k_batch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from src.trainers.inference import multivae_predict
from src.data.MultiVAE.MultiVAE_dataset import tensor_to_csr
from tqdm import tqdm


update_count = 0

def train(args, model, train_dataset, valid_dataset, logger, setting):
    
    # Optimizer and Scheduler 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optimizer_module, args.optimizer.type)(trainable_params,
                                                               **args.optimizer.args)
    
    if args.lr_scheduler.use:
        args.lr_scheduler.args = {k: v for k, v in args.lr_scheduler.args.items() 
                                  if k in getattr(scheduler_module, args.lr_scheduler.type).__init__.__code__.co_varnames}
        lr_scheduler = getattr(scheduler_module, args.lr_scheduler.type)(optimizer, 
                                                                         **args.lr_scheduler.args)
    else:
        lr_scheduler = None
    
    # wandb 설정
    if args.wandb:
            import wandb   
    
    best_r10 = -np.inf
    N = len(train_dataset)

    # loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False)
    
    global update_count

    for epoch in range(1, args.train.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
                
        for batch_idx, data in tqdm(enumerate(train_loader),
                                    desc=f'[Epoch {epoch:02d}/{args.train.epochs:02d}]'):
            
            data = data.squeeze(1).to(args.device) # (batch_size, p_dim)
            optimizer.zero_grad()

            if args.other_params.args.total_anneal_steps > 0:
                anneal = min(args.other_params.args.anneal_cap,
                            1. * update_count / args.other_params.args.total_anneal_steps)
            else:
                anneal = args.other_params.args.anneal_cap

            output = model(data)
            loss = multivae_loss(data, output, anneal)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            update_count += 1

            if batch_idx % args.other_params.args.log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.dataloader.batch_size)),
                        elapsed * 1000 / args.other_params.args.log_interval,
                        train_loss / args.other_params.args.log_interval))
                start_time = time.time()
                train_loss = 0.0

        if args.lr_scheduler.use and args.lr_scheduler.type != 'ReduceLROnPlateau':
            lr_scheduler.step()

        train_loss = train_loss / len(range(0, N, args.dataloader.batch_size))
        
        valid_loss, r10 = evaluate(args, model, valid_loader)
        if args.lr_scheduler.use and args.lr_scheduler.type == 'ReduceLROnPlateau':
                lr_scheduler.step(valid_loss)
                
    
        msg = ''
        msg += '| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | r10 {:5.3f}'.format(
                    epoch, time.time() - start_time, valid_loss, r10)
        print('-' * 89)
        print(msg)
        print('-' * 89)
        logger.log(epoch=epoch, train_loss=train_loss, valid_loss=valid_loss, valid_r10=r10)
        # wandb logging
        if args.wandb:
            wandb.log({'Epoch': epoch,
                       'Train Loss': train_loss,
                       'Valid Loss': valid_loss,
                       'Recall@10': r10})


        if args.train.save_best_model:
            if r10 > best_r10:
                best_r10 = r10
                os.makedirs(args.train.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt')
        else:
            os.makedirs(args.train.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_e{epoch:02}.pt')        
            
    logger.close()       
            
    return model




def evaluate(args, model, dataloader):
    # Turn on evaluation mode
    model.eval()

    # DataLoader 생성
    global update_count
    total_val_loss_list = []
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

            # 손실 함수 계산
            loss = multivae_loss(data_tr, output, anneal)
            total_val_loss_list.append(loss.item())

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

    return (np.nanmean(total_val_loss_list), 
            np.nanmean(r10_list))
    
    


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