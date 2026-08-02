import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from utils import generate_torch_loader_snippet

device = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_data(path):
    transform = v2.Compose([v2.Resize((128, 128)), v2.ToTensor(), v2.Normalize([0.5] * 3, [0.5] * 3)])
    dataset = datasets.ImageFolder(path, transform)
    targets = dataset.targets
    classes = len(dataset.classes)
    print(dataset.classes)
    train, test = train_test_split(range(len(dataset)), test_size=0.5, stratify=targets)

    dataloader = DataLoader(Subset(dataset, train), 32, shuffle=True)
    testloader = DataLoader(Subset(dataset, test), 32, shuffle=False)
    return classes, dataloader, testloader

class AgentCNN(nn.Module):

    def __init__(self, classes, kernel, pool, drop):
        super().__init__()
        
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 24, kernel),
            nn.MaxPool2d(pool),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(24, 24, kernel),
            nn.MaxPool2d(pool),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(drop),
        )

        self.fcnn_model = nn.Sequential(
            nn.Linear(self.get_flattened_size(), 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(drop),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, classes),
        )
    
    def get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 128, 128)
            x = self.cnn_model(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.cnn_model(x)
        x = torch.flatten(x, 1)
        x = self.fcnn_model(x)
        return x
    
def train_model(model, dataloader, epochs):
    losses = []
    optimiser = torch.optim.Adam(model.parameters())
    cel = nn.CrossEntropyLoss()

    model.train()
    for i in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            out = model(x_batch)
            loss = cel(out, y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)

    return model, losses
    
def eval_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred_do = torch.argmax(model(x), dim=1)
            correct += (pred_do == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test accuracy: {acc:.4f}")

def pipeline(datapath, parampath, outpath, funcpath):
    classes, dataloader, testloader = fetch_data(datapath)
    model = AgentCNN(classes, (4, 4), (4, 4), 0.1)
    model.to(device)
    if parampath: model.load_state_dict(torch.load(parampath))
    train_model(model, dataloader, 20)
    eval_model(model, testloader)
    torch.save(model.state_dict(), outpath)
    snippet = generate_torch_loader_snippet(model, prefer="auto", compression="zlib")
    with open(funcpath, "w") as f: f.writelines(snippet)

pipeline('Model/final_data', None, 'Model/params', 'Model/get_model')