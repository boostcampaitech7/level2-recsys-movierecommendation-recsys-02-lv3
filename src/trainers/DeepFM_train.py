import torch
import time
import numpy as np
import os
from src.loss.loss_fn import deepfm_loss
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score


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
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.test.batch_size, shuffle=False
    )

    global update_count

    best_epoch, best_auc, best_acc, early_stopping = 0, 0, 0, 0

    for epoch in range(1, args.train.epochs + 1):
        model.train()
        train_loss = batch_train_loss = 0.0
        start_time = time.time()

        for batch_idx, data in tqdm(
            enumerate(train_loader), desc=f"[Epoch {epoch:02d}/{args.train.epochs:02d}]"
        ):
            X, y = data
            optimizer.zero_grad()

            output = model(X)
            loss = deepfm_loss(output, y.float())

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

        train_loss = train_loss / len(range(0, N, args.dataloader.batch_size))

        AUC, ACC, RECALL, valid_loss = evaluate(args, model, valid_loader)

        msg = ""
        msg += "| end of epoch {:3d} | time: {:4.2f}s | AUC {:5.3f} | ACC {:5.3f} | RECALL {:5.3f} | train_loss {:.3f}".format(
            epoch, time.time() - start_time, AUC, ACC, RECALL * 0.1, train_loss
        )
        print("-" * 89)
        print(msg)
        print("-" * 89)
        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_r10=RECALL * 0.1,
        )
        # wandb logging
        if args.wandb:
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                    "AUC": AUC,
                    "ACC": ACC,
                    "RECALL": RECALL * 0.1,
                }
            )

        if args.train.save_best_model:
            if AUC > best_auc:
                best_epoch, best_auc, best_acc, early_stopping = epoch, AUC, ACC, 0
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
            early_stopping += 1
            if early_stopping == args.early_stopping:
                print("##########################")
                print(f"Early stopping triggered at epoch {epoch}")
                print(
                    f"BEST AUC: {best_auc}, ACC: {best_acc}, BEST EPOCH: {best_epoch}"
                )
                break

    logger.close()

    checkpoint = torch.load(
        f"{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt"
    )
    model.load_state_dict(checkpoint)

    return model


def evaluate(args, model, dataloader):
    # Turn on evaluation mode
    model.eval()

    # DataLoader 생성
    global update_count
    total_val_loss_list = []
    predicts = []
    targets = []

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            X, y = data

            # 모델 예측
            output = model(X)

            # 손실 함수 계산
            loss = deepfm_loss(output, y.float())
            total_val_loss_list.append(loss.item())

            predict_logits = output.detach().cpu()
            target = y.tolist()

            predict_probs = torch.sigmoid(predict_logits)

            predicts.extend(predict_probs.tolist())
            targets.extend(target)

    predicts = np.array(predicts)
    targets = np.array(targets)

    # 결과 평균 계산
    auc = roc_auc_score(targets, predicts)
    rounded_pred = np.rint(predicts)
    acc = accuracy_score(targets, rounded_pred)
    recall = recall_score(targets, rounded_pred)

    return auc, acc, recall, np.nanmean(total_val_loss_list)


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
        dataset, batch_size=args.test.batch_size, shuffle=False
    )

    AUC, ACC, RECALL, valid_loss = evaluate(args, model, test_loader)
    print("=" * 89)
    print(
        "| End of training | AUC {:5.3f} | ACC {:5.3f} | RECALL {:5.3f} | ".format(
            AUC, ACC, RECALL * 0.1
        )
    )
    print("=" * 89)

    return model
