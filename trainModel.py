import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from models import *
from datasets2 import *
import platform
import copy
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
from unet_model import UNet

TRAIN = 'train'
VAL = 'validation'

MASTER_RANK = 0
SAVE_INTERVAL =1

DATASET_PATH = '/s/chopin/b/grad/jhfitzg/cs535-term-project/data/next-day-wildfire-spread'
SAVE_MODEL_PATH = '/s/chopin/b/grad/jhfitzg/cs535-term-project/savedModels'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master', default='mercury',
                        help='master node')
    parser.add_argument('-p', '--port', default='30437',
                         help = 'master node')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int,
                        metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    print(f'initializing ddp: GLOBAL_RANK: {args.nr}, MEMBER: {int(args.nr)+1} / {args.nodes}')
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    os.environ['MASTER_ADDR'] = args.master              #
    os.environ['MASTER_PORT'] = args.port                  #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################



def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    validate = True
    print("Current GPU", gpu,"\n RANK: ",rank)
    batch_size = 100
    # Data loading code

    datasets = {
        TRAIN: AugmentedWildfireDataset(
            f"{DATASET_PATH}/{TRAIN}.data",
            f"{DATASET_PATH}/{TRAIN}.labels",
        ),
        VAL: WildfireDataset(
            f"{DATASET_PATH}/{VAL}.data",
            f"{DATASET_PATH}/{VAL}.labels",
        )
    }

    samplers = {
        TRAIN: torch.utils.data.distributed.DistributedSampler(
            datasets[TRAIN],
            num_replicas=args.world_size,
            rank=rank
        )
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(
            dataset=datasets[TRAIN],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=samplers[TRAIN]
        ),
        VAL: torch.utils.data.DataLoader(
            dataset=datasets[VAL],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    }

    torch.manual_seed(0)

    #model = LogisticRegression(12288, 1024)
    #model = BinaryClassifierCNN(32)
    #model = ConvolutionalAutoencoder()
    model = UNet(12, 1, True)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    #criterion = nn.BCELoss().cuda(gpu)
    criterion = nn.BCEWithLogitsLoss().cuda(gpu) # This is for UNet
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])

    start = datetime.now()
    total_step = len(dataLoaders[TRAIN])
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')

    best_avg_loss_val = float("inf")
    best_avg_acc_val = -float("inf")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        loss_train = 0
        total_samples_seen = 0
        for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            total_samples_seen += len(images)

            # Forward pass
            outputs = model(images)

            # Not entirely sure if this flattening is required or not
            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)
            loss = criterion(outputs, labels)

            loss_train += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Samples processed {}, Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i,
                    total_step,
                    total_samples_seen,
                    loss.item())
                )

        train_loss_history.append(loss_train / len(dataLoaders[TRAIN]))
    
        if validate:
            loss_val = 0
            acc_val = 0
            total_pixels = 0
            
            val_loader = dataLoaders[VAL]
            for i, (images, labels) in enumerate(val_loader):
                k = len(val_loader) // 4

                # if i % k == 0:
                #         print("\rValidation batch {}/{}".format(i, len(val_loader)))

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # Forward pass
                outputs = model(images)

                labels = torch.flatten(labels)
                outputs = torch.flatten(outputs)

                # if probability > 0.5 then fire is predicted, otherwise no fire
                preds = torch.round(torch.sigmoid(outputs)) # UNet output isn't going through sigmoid, loss func handles it
                #preds = torch.round(outputs)

                loss = criterion(outputs, labels)

                loss_val += loss.item() # batch loss
                acc_val += torch.sum(preds == labels.data)
                total_pixels += len(labels)
            
            curr_avg_loss_val = loss_val / len(val_loader)
            curr_avg_acc_val = 100 * acc_val / total_pixels

            val_loss_history.append(curr_avg_loss_val)

            print(f"Average validation batch loss = {curr_avg_loss_val}")
            print(f"Validation acc = {curr_avg_acc_val}%")

            if(best_avg_loss_val > curr_avg_loss_val and epoch % SAVE_INTERVAL == 0):
                # save model
                print("Saving model...")
                best_avg_loss_val = curr_avg_loss_val
                filename = f'model-{model.module.__class__.__name__}-bestLoss-Rank-{rank}.weights'
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/{filename}')
                print("Model has been saved!")
            else:
                print("Model is not being saved")

    print("Reached end of train function")

    with open(f'{SAVE_MODEL_PATH}/model-{model.module.__class__.__name__}-train-loss-Rank-{rank}.history', 'wb') as handle:
        pickle.dump(train_loss_history, handle)
    print("Successfully pickled the training loss history")

    with open(f'{SAVE_MODEL_PATH}/model-{model.module.__class__.__name__}-validation-loss-Rank-{rank}.history', 'wb') as handle:
        pickle.dump(val_loss_history, handle)
    print("Successfully pickled the validation loss history")
        
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        print(f"Endtime: {datetime.now()}")
    

if __name__ == '__main__':
    main()

