from sklearn.manifold import TSNE
import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import umap
from tqdm import tqdm
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import torchvision.datasets as datasets
from siamese_network import SiameseNetwork

def obtainResnet18featureExtractor():
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    # remove the classifier head, leave only feature extractor
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    # add flatten layer to the feature extractor,
    # so shape is (n_samples, n_features) 
    feature_extractor = nn.Sequential(feature_extractor, nn.Flatten())
    
    return feature_extractor

def getEmbeddings(model, loader, device):
    allEmbeddings = None
    allLabels = None

    model.eval()
    with torch.no_grad():

        for data, labels in tqdm(loader):
            data = data.to(device)
            #output = model.forward_once(data)
            output = model(data)
            
            if allEmbeddings is None:
                allEmbeddings = output
                allLabels = labels
            else:
                allEmbeddings = torch.vstack((allEmbeddings, output))
                allLabels = torch.cat((allLabels, labels))
                
    return allEmbeddings, allLabels
            
def visualizeDataTSNE(embeddings, labels, labelsNames):
    # Transform to 2D
    tsne = TSNE()
    transEmbeddings = tsne.fit_transform(embeddings)
    
    # Create dataframe
    tsne_result_df = pd.DataFrame({'tsne_1': transEmbeddings[:,0], 'tsne_2': transEmbeddings[:,1], 'Class': labelsNames})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='Class', data=tsne_result_df, palette="deep", ax=ax,s=5, linewidth=0)
    lim = (transEmbeddings.min()-5, transEmbeddings.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.get_legend().remove()
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title("t-SNE visualization")
    plt.show()
    
def visualizeDataUMAP(embeddings, labels, labelsNames):
    # Fit UMAP to processed data
    manifold = umap.UMAP().fit(embeddings, labels)
    transEmbeddings = manifold.transform(embeddings)
    
    # Create dataframe
    tsne_result_df = pd.DataFrame({'tsne_1': transEmbeddings[:,0], 'tsne_2': transEmbeddings[:,1], 'Class': labelsNames})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='Class', data=tsne_result_df, palette="deep", ax=ax,s=5, linewidth=0)
    lim = (transEmbeddings.min()-5, transEmbeddings.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.get_legend().remove()
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title("UMAP visualization")
    plt.show()


def visualizeDataPCA(embeddings, labels, labelsNames):
    # Fit UMAP to processed data
    pca = PCA(n_components = 2)
    transEmbeddings = pca.fit_transform(embeddings, labels)
    
    # Create dataframe
    tsne_result_df = pd.DataFrame({'tsne_1': transEmbeddings[:,0], 'tsne_2': transEmbeddings[:,1], 'Class': labelsNames})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='Class', data=tsne_result_df, palette="deep", ax=ax,s=20)
    lim = (transEmbeddings.min()-5, transEmbeddings.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title("PCA visualization")
    plt.show()

if __name__ == "__main__":
    
    device = "cuda"
    batch_size = 512
    trained = True
    weightsPath = "resnet18_sm_constrastive_3_car_lr_0.0001_batchSize_128_miner_no.pth"
    #weightsPath = "sm_constrastive_car.pt"
    
    # Get model
    model = obtainResnet18featureExtractor()
    #model = SiameseNetwork()
    model = model.to(device)
    
    # Load weights if it is trained
    if trained:
        model.load_state_dict(torch.load(weightsPath, map_location=device))
    
    # Resize the images and transform to tensors
    transformation_images = ResNet18_Weights.IMAGENET1K_V1.transforms()
    # transformation_images = transform=transforms.Compose([transforms.Resize((100,100)),
    #                                                       transforms.ToTensor(),
    #                                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

                                                
    # Load the test dataset
    test_dataset = datasets.ImageFolder(root="./train_dataset/",transform=transformation_images)
    
    # Create a simple dataloader just for simple visualization
    test_dataloader = DataLoader(test_dataset,
                            shuffle=False,
                            num_workers=0,
                            batch_size=batch_size)
    
    # Get embeddings
    embeddings, labels = getEmbeddings(model, test_dataloader, device)
    
    # Pass to numpy arrays
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    #labelsStr = labelsName[labels]
    
    # Visualize T-SNE
    visualizeDataTSNE(embeddings, labels, labels)
    
    # Visualize UMAP
    visualizeDataUMAP(embeddings, labels, labels)
    
    # Visualize PCA
    #visualizeDataPCA(embeddings, labels, labelsStr)
    