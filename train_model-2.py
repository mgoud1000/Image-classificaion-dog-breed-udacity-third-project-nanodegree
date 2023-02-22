#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import logging
import os
import sys
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

def test(model, test_loader,criterion,device):
    """
    Defining model testing function that will be applied at testing dataset
    """
    print("Testing model on the whole testing dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    hook = get_hook(create_if_not_exists = True)
    if hook:
        hook.set_mode(modes.EVAL)
    for inputs,labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        _, preds=torch.max(outputs,1) # get the index of the max log-probability
        running_loss+=loss.item()*inputs.size(0)# sum up batch loss
        running_corrects+=torch.sum(preds==labels.data).item()
    total_loss=running_loss/len(test_loader.dataset)
    total_acc=running_corrects/len(test_loader.dataset)
    logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))
    
def train(model, train_loader, validation_loader,criterion, optimizer,device,epochs):
    best_loss=1e6
    image_dataset={'train':train_loader,'valid':validation_loader}
    loss_counter=0
    hook = get_hook(create_if_not_exists = True)
    if hook:
        hook.set_mode(modes.TRAIN)                 
    for epoch in range(1,epochs+1):
        logger.info(f"Epoch: {epoch} - Training Model on Complete Training Dataset")
        for phase in ['train','valid']:
            logger.info(f"Training on Epoch {epoch} and Phase {phase}")
            running_loss=0
            running_corrects=0
            running_samples=0
            model.train() if phase=='train' else model.eval()
            for _,(inputs,labels) in enumerate (image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                if phase=='train':
                    hook = get_hook(create_if_not_exists = True)
                    if hook:
                        hook.set_mode(modes.TRAIN)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _,preds=torch.max(outputs,1)
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data).item()
                running_samples += len(inputs)
                
                if running_samples % 1000==0:
                    accuracy=running_corrects/running_samples
                    logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy))
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            logger.info(f"Epoch {epoch} loss = {epoch_loss} and Accuracy = {epoch_acc}")
            if phase=='valid':
                hook = get_hook(create_if_not_exists = True)
                if hook:
                    hook.set_mode(modes.EVAL)
                if epoch_loss>best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

            total_loss = running_loss / len(train_loader.dataset)
            total_acc = running_corrects/ len(image_dataset[phase].dataset)
            logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                        total_loss, running_corrects,
                        len(image_dataset[phase].dataset),
                        100.0 * total_acc))
        if loss_counter == 1:
            break
    return model

    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def create_data_loaders(data, batch_size):
    train_dataset_path = os.path.join(data, "train")
    validation_dataset_path=os.path.join(data,"valid")
    test_dataset_path = os.path.join(data, "test")


    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])

    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)
    valid_dataset=torchvision.datasets.ImageFolder(root=validation_dataset_path, transform=training_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, validation_loader


def main(args):
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    model=model.to(device)
    optimizer=optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion=nn.CrossEntropyLoss()
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size )
    hook = get_hook(create_if_not_exists = True)
    if hook:
        hook.register_module(model)
        hook.register_loss(criterion)
        
    logger.info(f"Running on Device {device}")
    logger.info(f"Hyperparameters : LR: {args.lr}, Weight-decay: {args.weight_decay}, Batch Size: {args.batch_size} ")
    logger.info(f"Data Dir Path: {args.data_dir}")
    logger.info(f"Model Dir  Path: {args.model_dir}")
    logger.info(f"Output Dir  Path: {args.output_dir}")
    logger.info("Starting to Save the Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Completed Saving the Model")
    #Adding in the epoch to train and test/validate for the same epoch at the same time.
    epochs=2
    model=train(model, train_loader, validation_loader, criterion, optimizer, device,epochs)
    test(model, test_loader, criterion, device)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    Adding all the hyperparameters needed to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY",
                        help="weight decay coefficient (default 1e-2)")
    parser.add_argument("--test_batch_size",type=int,default=256,metavar="N",
                        help="input batch size for testing (default: 500)")

    # Using sagemaker OS Environ's channels to locate training data, model dir and output dir to save in S3 bucket
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args = parser.parse_args()

    main(args)