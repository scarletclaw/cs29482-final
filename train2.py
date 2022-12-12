"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
torchvision
"""
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
from torch import nn
from models.resnet2 import *
from torchvision import datasets, transforms

import logging
from datetime import datetime
logger = logging.getLogger()

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/100/Imagenet Generate Group Info')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--comment', type=str, default="test")

parser.add_argument('--final_dim', default=512, type=int)
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on testing set')
args = parser.parse_args()

logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

comment = "{}_{}_{}".format(str(datetime.now().strftime(r'%m%d_%H%M%S')), args.final_dim, args.comment)
resultDirPath = Path("log") / comment
resultDirPath.mkdir(parents=True, exist_ok=True)

fileHandler = logging.FileHandler(resultDirPath / "info.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.01               # learning rate

logger.info("Epoch: {} | batch size: {} | LR: {} | Model: {}".format(EPOCH, BATCH_SIZE, LR, args.arch))

train_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = datasets.CIFAR10(root='./data', train=True,
        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=args.workers)

testset = datasets.CIFAR10(root='./data', train=False,
        download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=args.workers)

if args.arch == 'resnet18':
    model = ResNet18(args.final_dim)
elif args.arch == 'vgg11':
    model = VGG('VGG11', args.final_dim)
else:
    raise NotImplementedError("Not support {}".format(args.arch))

model = model.cuda()
logger.info(model)

# load from pre-trained model (only cnn part)
model.load_state_dict(torch.load(r'/content/drive/MyDrive/ColabNotebooks/CS294-082/models/best_checkpoint.pth.tar'), strict=False)
# model.load_state_dict(torch.load(r'C:\Users\scarl\OneDrive\桌面\UCB MEng EECS\CS294-082\cs29482-final-master\log\1206_215030_512_test\best_checkpoint.pth.tar'), strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

if  args.evaluate:
    y_pred = []
    y_true = []
    with torch.no_grad():
        trange = tqdm(enumerate(test_loader), total=len(test_loader), desc="Train|Epoch {}".format(epoch))
        for step, (inputs, labels) in trange:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)                             # model output
            loss = loss_func(outputs, labels)                   # cross entropy loss

            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted)
            y_true.append(labels)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_loss += loss.item() * labels.size(0)
            trange.set_postfix(loss=loss.item(), Acc=test_correct / test_total)
    epoch_test_loss = test_loss / test_total
    epoch_test_acc = test_correct / test_total
else:

    # training and testing
    best_test_loss = 1e10
    for epoch in range(EPOCH):
        # training 
        model.train()
        trange = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train|Epoch {}".format(epoch))
        train_total = 0
        train_correct = 0
        train_loss = 0
        for step, (inputs, labels) in trange:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)                         # model output
            loss = loss_func(outputs, labels)               # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * labels.size(0)
            trange.set_postfix(loss=loss.item(), Acc=train_correct / train_total)
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        test_total = 0
        test_correct = 0
        test_loss = 0
        with torch.no_grad():
            trange = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test|Epoch {}".format(epoch))
            for step, (inputs, labels) in trange:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)                             # model output
                loss = loss_func(outputs, labels)                   # cross entropy loss

                _, predicted = torch.max(outputs, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += loss.item() * labels.size(0)
                trange.set_postfix(loss=loss.item(), Acc=test_correct / test_total)
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = test_correct / test_total

        logger.info('Epoch: {}'.format(epoch))
        logger.info('Train | loss: {:.4f} | accuracy {:.2f}'.format(epoch_train_loss, epoch_train_acc))
        logger.info('Test | loss: {:.4f} | accuracy {:.2f}'.format(epoch_test_loss, epoch_test_acc))

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            filename = resultDirPath / "best_checkpoint.pth.tar"
            torch.save(model.state_dict(), filename)
            logger.info("Current Best(loss: {:.4f}, acc: {:.2f}) Save to: {}".format(epoch_test_loss, epoch_test_acc, filename))

    
