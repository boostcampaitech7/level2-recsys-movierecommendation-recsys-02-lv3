import torch
import time
import numpy as np
import os
from src.loss.loss_fn import deepfm_loss
from src.utils.metrics import Recall_at_k_batch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from src.trainers.inference import multivae_predict
from src.data.MultiVAE.MultiVAE_dataset import tensor_to_csr
from tqdm import tqdm
from sklearn.metrics import f1_score


update_count = 0


def train(args, model, train_dataset, valid_dataset, logger, setting):

    # Optimizer and Scheduler 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optimizer_module, args.optimizer.type)(
        trainable_params, **args.optimizer.args
    )

    if args.lr_scheduler.use:
        args.lr_scheduler.args = {
            k: v
            for k, v in args.lr_scheduler.args.items()
            if k
            in getattr(
                scheduler_module, args.lr_scheduler.type
            ).__init__.__code__.co_varnames
        }
        lr_scheduler = getattr(scheduler_module, args.lr_scheduler.type)(
            optimizer, **args.lr_scheduler.args
        )
    else:
        lr_scheduler = None

    # wandb 설정
    if args.wandb:
        import wandb

    best_r10 = -np.inf
    N = len(train_dataset)

    # loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=args.dataloader.shuffle,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False
    )

    global update_count

    for epoch in range(1, args.train.epochs + 1):
        model.train()
        train_loss = batch_train_loss = 0.0
        start_time = time.time()

        for batch_idx, data in tqdm(
            enumerate(train_loader), desc=f"[Epoch {epoch:02d}/{args.train.epochs:02d}]"
        ):
            X, y = data
            X = [x.to(args.device) for x in X]
            y = y.to(args.device)
            optimizer.zero_grad()

            output = model(X)
            loss = deepfm_loss(y, output)

            loss.backward()
            train_loss += loss.item()
            batch_train_loss += loss.item()
            optimizer.step()
            update_count += 1

            if batch_idx % args.other_params.args.log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | loss {:4.2f}".format(
                        epoch,
                        batch_idx + 1,
                        len(range(0, N, args.dataloader.batch_size)),
                        elapsed * 1000 / args.other_params.args.log_interval,
                        batch_train_loss / args.other_params.args.log_interval,
                    )
                )
                start_time = time.time()
                batch_train_loss = 0.0

        if args.lr_scheduler.use and args.lr_scheduler.type != "ReduceLROnPlateau":
            lr_scheduler.step()

        train_loss = train_loss / len(range(0, N, args.dataloader.batch_size))

        valid_loss, avg_f1 = evaluate(args, model, valid_loader)
        if args.lr_scheduler.use and args.lr_scheduler.type == "ReduceLROnPlateau":
            lr_scheduler.step(valid_loss)

        msg = ""
        msg += "| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | f1 {:5.3f}".format(
            epoch, time.time() - start_time, valid_loss, avg_f1
        )
        print("-" * 89)
        print(msg)
        print("-" * 89)
        logger.log(
            epoch=epoch, train_loss=train_loss, valid_loss=valid_loss, valid_f1=avg_f1
        )
        # wandb logging
        if args.wandb:
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                    "F1": avg_f1,
                }
            )

        if args.train.save_best_model:
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                os.makedirs(args.train.ckpt_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt",
                )
        else:
            os.makedirs(args.train.ckpt_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{args.train.ckpt_dir}/{setting.save_time}_{args.model}_e{epoch:02}.pt",
            )

    logger.close()

    return model


def evaluate(args, model, dataloader):
    # Turn on evaluation mode
    model.eval()

    # DataLoader 생성
    global update_count
    total_val_loss_list = []
    f1_scores = []  # F1 score를 저장할 리스트

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            X, y = data
            X = [x.to(args.device) for x in X]
            y = y.to(args.device)

            # 모델 예측
            output = model(X)

            # 손실 함수 계산
            loss = deepfm_loss(y, output)
            total_val_loss_list.append(loss.item())

            # 예측값을 0 또는 1로 변환 (확률값을 이진 클래스 예측으로 변환)
            predicted = torch.sigmoid(output)  # sigmoid를 통해 확률값으로 변환
            predicted_class = (predicted >= 0.5).float()  # 0.5 이상이면 1, 아니면 0

            # F1 score 계산 (y와 predicted_class 비교)
            f1 = f1_score(y.cpu().numpy(), predicted_class.cpu().numpy())
            f1_scores.append(f1)

    # 결과 평균 계산
    avg_f1 = np.nanmean(f1_scores)
    avg_loss = np.nanmean(total_val_loss_list)

    return avg_loss, avg_f1


def test(args, model, dataset, setting, checkpoint=None):

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        if args.train.save_best_model:
            model_path = (
                f"{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt"
            )
        else:
            # best가 아닐 경우 마지막 에폭으로 테스트하도록 함
            model_path = f"{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{args.train.epochs-1:02d}.pt"
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # Run on test data

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.dataloader.batch_size, shuffle=False
    )

    test_loss, r10 = evaluate(args, model, test_loader)
    print("=" * 89)
    print(
        "| End of training | test loss {:4.2f} |  r10 {:4.2f} | ".format(test_loss, r10)
    )
    print("=" * 89)

    return model
