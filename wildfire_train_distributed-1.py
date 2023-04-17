import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp
import platform
import random
import numpy as np
import pickle
from torch.utils.data import Dataset, IterableDataset, DataLoader


MASTER_RANK = 0

# Neural networks that can be used on the next day wildfire spread dataset
# Make sure the training data is scrubbed of any target fire masks that have missing data

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.linear(x))
        x = x.reshape(-1, 1, 32, 32)
        return x


class BinaryClassifierCNN(torch.nn.Module):
    def __init__(self, image_size):
        flattened_conv2_output_dimensions = (image_size//4)**2
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 16, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(32 * flattened_conv2_output_dimensions, 1024), # 1024 pixels, output represents probability of fire.
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        x = x.reshape(-1, 1, 32, 32)
        return x


# Tutorial for the autoencoder: https://www.youtube.com/watch?v=345wRyqKkQ0
class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    
    def forward(self, x):
        return x.view(self.shape)


class Trim(torch.nn.Module):
    def __init__(self, *args):
        super(Trim, self).__init__()
    
    def forward(self, x):
        return x[:, :, :32, :32]


class ConvolutionalAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(12, 16, 3, 1, 0), # 32 x 32 -> 30 x 30
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(16, 32, 3, 2, 0), # 30 x 30 -> 14 x 14
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(32, 32, 3, 2, 0), # 14 x 14 -> 6 x 6
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 2) # 1152 = 32 * 6  * 6
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 1152),
            Reshape(-1, 32, 6, 6),
            torch.nn.ConvTranspose2d(32, 32, 3, 1, 0), # 6 x 6 -> 8 x 8
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(32, 16, 3, 2, 1), # 8 x 8 -> 15 x 15
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(16, 16, 3, 2, 0), # 15 x 15 -> 31 x 31
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(16, 1, 3, 1, 0), # 31 x 31 -> 33 x 33
            Trim(),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master', default='mars',
                        help='master node')
    parser.add_argument('-p', '--port', default='30435',
                         help = 'master node')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=15, type=int,
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


def unpickle(f):
    with open(f, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# Used for testing and validation datasets
class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, labels_filename, transform=None):
        self.data = unpickle(data_filename)
        self.labels = unpickle(labels_filename)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (torch.from_numpy(self.data[idx]), torch.from_numpy(np.expand_dims(self.labels[idx], axis=0)))

        return sample

# Used for the training dataset
class OversampledWildfireDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, labels_filename, transform=None):
        self.data = unpickle(data_filename)
        self.labels = unpickle(labels_filename)
        self.oversample_indices = []
        
        for i in range(len(self.data)):
            unique_target, counts_target = np.unique(self.labels[i], return_counts=True)
            target_counts_map = {int(unique_target[i]): int(counts_target[i]) for i in range(len(unique_target))}
            
            for key, value in target_counts_map.items():
                if key == 1 and (value/1024) >= 0.15: # adjust the fraction to see which masks have a lot of fire
                    self.oversample_indices.append(i)

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        idx = idx if idx < len(self.data)//2 else self._get_random_oversample_index()

        sample = (torch.from_numpy(self.data[idx]), torch.from_numpy(np.expand_dims(self.labels[idx], axis=0)))
        
        return sample
    
    def _get_random_oversample_index(self):
        return random.choice(self.oversample_indices)

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    print("Current GPU", gpu,"\n RANK: ",rank)
    batch_size = 100
    # Data loading code
    train_dataset = OversampledWildfireDataset('./data/next-day-wildfire-spread/train.data', './data/next-day-wildfire-spread/train.labels')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    torch.manual_seed(0)

    #model = LogisticRegression(12288, 1024)
    #model = BinaryClassifierCNN(32)
    model = ConvolutionalAutoencoder()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    criterion = nn.BCELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])

    start = datetime.now()
    total_step = len(train_loader)
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')
    for epoch in range(args.epochs):
        total_samples_seen = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            total_samples_seen += len(images)
            # Forward pass
            outputs = model(images)
            # Not entirely sure if this flattening is required or not
            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Samples Processed: {}, Loss: {:.4f}'.format(
            epoch + 1,
            args.epochs,
            total_samples_seen,
            loss.item())
        )
        # if validate:
        #         for i, (images, labels) in enumerate(val_loader):
        #             if i % 100 == 0:
        #                     print("\rValidation batch {}/{}".format(i, len(val_loader)), end='', flush=True)
        #             images = images.cuda(non_blocking=True)
        #             labels = labels.cuda(non_blocking=True)
        #             # Forward pass
        #             outputs = model(images)
        #             loss = criterion(outputs, labels)
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        print(f"Endtime: {datetime.now()}")


if __name__ == '__main__':
    main()

