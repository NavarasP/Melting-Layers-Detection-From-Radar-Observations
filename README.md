Notebook usage (example):

# 1) Imports
from io_gpm import list_hdf5_files
from dataset_gpm import make_patch_index, get_patch_as_tensor
from model_unet import UNet
from loss_metrics import ComboLoss
from train_utils import train_one_epoch, evaluate
from viz import visualize_prediction
import torch
from torch.utils.data import DataLoader, Dataset

# 2) Build an index of patches
files = list_hdf5_files('GPM_DPR_2024')
index = make_patch_index(files, patch_size=128, step=128)  # returns list of dicts

# 3) Create a small custom Dataset wrapper (example)
class PatchDataset(Dataset):
    def __init__(self, index):
        self.index = index
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        meta = self.index[i]
        X, mask = get_patch_as_tensor(meta['file'], meta['scan_start'], patch_size=128, device='cpu')
        return X, mask

ds = PatchDataset(index[:100])  # small subset
dl = DataLoader(ds, batch_size=4, shuffle=True)

# 4) Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = ComboLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=5e-3)

# 5) Train one epoch
train_loss = train_one_epoch(model, dl, opt, criterion, device)

# 6) Visualize a sample
X, y = ds[0]
visualize_prediction(model, X, y, device=device, mc_dropout=False)
