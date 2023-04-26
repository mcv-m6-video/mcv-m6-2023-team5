import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import wandb
import numpy as np
# Creating some helper functions

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        if mining_func is None:
            loss = loss_func(embeddings, labels)
        else:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )

            )
            #print("Num: ", mining_func.num_triplets)
            wandb.log({"iteration": batch_idx + (epoch-1)*len(train_loader), "loss": loss.item()})


def test(model, loss_func, device, test_loader):

    model.eval()
    losses = []
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            embeddings = model(data)

        loss = loss_func(embeddings, labels)

        losses.append(loss.item())
    print(
        "Test loss = {}".format(
            np.array(losses).mean()
        )

    )
    
    wandb.log({"test_loss": np.array(losses).mean()})
    return np.array(losses).mean()

if __name__ == "__main__":
    # Device
    device = torch.device("cuda")
    
    # Default transforms
    transformation_images =transforms.Compose([ResNet18_Weights.IMAGENET1K_V1.transforms(),
                                               #transforms.RandomResizedCrop(56, scale=(0.5, 1)),
                                               #transforms.ColorJitter(hue=.2, saturation=.2),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(20)])
    transformation_images_val =transforms.Compose([ResNet18_Weights.IMAGENET1K_V1.transforms(),
                                               #transforms.Resize(56)
                                               ])
    
    # Init datasets
    dataset_train = datasets.ImageFolder(root="./train_dataset/",transform=transformation_images)
    dataset_test = datasets.ImageFolder(root="./val_dataset/",transform=transformation_images_val)
    
    modelName = "resnet18"
    
    
    # Define hyperparameters
    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes =   [16, 32, 64, 128]
    minerOpts = ["no", "hard", "no", "semihard"]
    num_epochs = 50#100
    batch_size = 128#128
    lr = 1e-4
    minerOpt = "no"
    margin = 1
    
    for minerOpt in minerOpts:

        # Init model
        model1 = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = torch.nn.Sequential(*(list(model1.children())[:-1]))
        model = nn.Sequential(feature_extractor, nn.Flatten()).to(device)
        
        run = wandb.init(project='M6_WEEK5', job_type='train')
        wandb.run.name = modelName + "_siamese_lr_" + str(lr) + "_batchSize_" + str(batch_size) + "_miner_" + minerOpt

        print("Learning rate: ", lr)
        print("Batch size: ", batch_size)
        print("Miner: ", minerOpt)

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers = 0
        )
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers = 0)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        ### pytorch-metric-learning stuff ###
        distance = distances.LpDistance(power=2)
        #loss_func = losses.ContrastiveLoss(distance = distance)
        loss_func = losses.TripletMarginLoss(distance = distance, margin = margin)
        
        if minerOpt == "no":
            mining_func = None
        else:
            # mining_func = miners.BatchEasyHardMiner(
            #     neg_strategy=minerOpt,
            #     pos_strategy=minerOpt
            # )
            mining_func = miners.TripletMarginMiner(margin=margin/100, type_of_triplets=minerOpt)
        best_val = 100
        ### pytorch-metric-learning stuff ###
        for epoch in range(1, num_epochs + 1):
            train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        
            val_loss = test(model, loss_func, device, test_loader)
            if val_loss < best_val:
                torch.save(model.state_dict(), modelName + "_sm_constrastive_3_car_lr_" + str(lr) + "_batchSize_" + str(batch_size) + "_miner_" + minerOpt + ".pth")
                best_val = val_loss
        wandb.finish()
