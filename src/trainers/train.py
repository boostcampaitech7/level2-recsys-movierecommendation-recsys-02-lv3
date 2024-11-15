import torch
import time
import bottleneck as bn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import wandb
from src.loss.loss_fn import loss_function_vae
from src.utils.metrics import Recall_at_k_batch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module


def run(args, model, loader, setting):
    
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
    
    
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    current_dir = os.path.join(os.getcwd(), 'choi/level2-recsys-movierecommendation-recsys-02-lv3/')
    save_dir = os.path.join(current_dir, args.train.ckpt_dir)
    model_path = os.path.join(save_dir, setting.get_submit_filename(args)[9:-4]+'.pt')
    
    
    train_data = loader.load_data('train', False)
    valid_data_tr, valid_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')
    
    N = train_data.shape[0]
    idxlist = list(range(N))

    best_r10 = -np.inf
    global update_count
    update_count = 0

    for epoch in range(1, args.train.epochs + 1):
        epoch_start_time = time.time()
        train(args, model, optimizer, train_data, idxlist, N)
        val_loss, r10 = evaluate(args, model, valid_data_tr, valid_data_te, N)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | r10 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    r10))
        print('-' * 89)
        # wandb.log(dict(epoch=epoch,
        #                train_loss=train_loss,
        #                valid_loss=val_loss,
        #                r10=r10))


        # Save the model if the r10 is the best we've seen so far.
        if r10 > best_r10:
            with open(model_path, 'wb') as f:
                torch.save(model, f)
            best_r10 = r10
    
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, r10 = evaluate(args, model, test_data_tr, test_data_te, N)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} |  r10 {:4.2f} | '.format(test_loss, r10))
    print('=' * 89)
    return model



def train(args, model, optimizer, train_data, idxlist, N):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)

    for batch_idx, start_idx in enumerate(range(0, N, args.dataloader.batch_size)):
        end_idx = min(start_idx + args.dataloader.batch_size, N)

        # 이미 텐서로 변환된 데이터를 사용
        data = train_data[idxlist[start_idx:end_idx]].to(args.device)
        optimizer.zero_grad()

        if args.other_params.args.total_anneal_steps > 0:
            anneal = min(args.other_params.args.anneal_cap,
                        1. * update_count / args.other_params.args.total_anneal_steps)
        else:
            anneal = args.other_params.args.anneal_cap

        recon_batch, mu, logvar = model(data)
        loss = loss_function_vae(recon_batch, data, mu, logvar, anneal)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        update_count += 1

        if batch_idx % args.other_params.args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                  'loss {:4.2f}'.format(
                      args.epoch, batch_idx, len(range(0, N, args.dataloader.batch_size)),
                      elapsed * 1000 / args.other_params.args.log_interval,
                      train_loss / args.other_params.args.log_interval))
            start_time = time.time()
            train_loss = 0.0



def evaluate(args, model, data_tr, data_te, N):
    # Turn on evaluation mode
    model.eval()
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    total_val_loss_list = []
    r10_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, args.dataloader.batch_size):
            end_idx = min(start_idx + args.dataloader.batch_size, N)

            # 이미 텐서로 변환된 데이터를 사용
            data = data_tr[e_idxlist[start_idx:end_idx]].to(args.device)
            heldout_data = data_te[e_idxlist[start_idx:end_idx]].to(args.device)

            if args.other_params.args.total_anneal_steps > 0:
                anneal = min(args.other_params.args.anneal_cap,
                             1. * update_count / args.other_params.args.total_anneal_steps)
            else:
                anneal = args.other_params.args.anneal_cap

            # 모델 예측
            recon_batch, mu, logvar = model(data)

            # 손실 함수 계산
            loss = loss_function_vae(recon_batch, data, mu, logvar, anneal)
            total_val_loss_list.append(loss.item())

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            data = data.cpu().numpy()
            heldout_data = heldout_data.cpu().numpy()

            # Training data에서 이미 평가에 포함된 항목 제외
            recon_batch[data.nonzero()] = -np.inf

            # 평가 지표 계산
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r10_list.append(r10)

    # 결과 평균 계산
    r10_list = np.concatenate(r10_list)

    return (np.nanmean(total_val_loss_list), 
            np.nanmean(r10_list))