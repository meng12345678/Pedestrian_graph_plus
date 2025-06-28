import torch
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from sklearn.metrics import balanced_accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from jaad_dataloader23 import DataSet
from models.ped_graph23 import pedMondel

from pathlib import Path
import argparse
import os
import numpy as np


def seed_everything(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LitPedGraph(pl.LightningModule):
    def __init__(self, args, len_tr):
        super(LitPedGraph, self).__init__()
        self.balance = args.balance
        self.total_steps = len_tr * args.epochs
        self.lr = args.lr
        self.epochs = args.epochs
        self.ch = 4 if args.H3D else 3
        self.ch1, self.ch2 = 32, 64
        self.frames = args.frames
        self.velocity = args.velocity
        self.time_crop = args.time_crop

        self.model = pedMondel(args.frames, args.velocity, seg=args.seg, h3d=args.H3D, n_clss=3)

        # 权重设置，可调节 alpha 比例
        self.alpha = torch.tensor([0.7, 1.0, 0.5])
        self.criterion = FocalLoss(alpha=self.alpha, gamma=2.0)

    def forward(self, kp, f, v):
        return self.model(kp, f, v)

    def training_step(self, batch, batch_nb):
        x, y, f, v = batch[0], batch[1], batch[2] if self.frames else None, batch[3] if self.velocity else None

        if torch.rand(1).item() > 0.5 and self.time_crop:
            crop_size = torch.randint(2, 21, (1,)).item()
            x = x[:, :, -crop_size:]

        logits = self(x, f, v)
        loss = self.criterion(logits, y.view(-1).long())

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds, y.view(-1).long(), task='multiclass', num_classes=3)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc * 100.0, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y, f, v = batch[0], batch[1], batch[2] if self.frames else None, batch[3] if self.velocity else None

        logits = self(x, f, v)
        loss = self.criterion(logits, y.view(-1).long())

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds, y.view(-1).long(), task='multiclass', num_classes=3)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc * 100.0, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y, f, v = batch[0], batch[1], batch[2] if self.frames else None, batch[3] if self.velocity else None

        logits = self(x, f, v)
        loss = self.criterion(logits, y.view(-1).long())

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds, y.view(-1).long(), task='multiclass', num_classes=3)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc * 100.0, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optm = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        lr_scheduler = {
            'name': 'OneCycleLR',
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optm, max_lr=self.lr, div_factor=10.0, final_div_factor=1e4, total_steps=self.total_steps, verbose=False),
            'interval': 'step', 'frequency': 1,
        }
        return [optm], [lr_scheduler]

def data_loader(args):
    transform = A.Compose([
        A.ToPILImage(),
        A.RandomPosterize(bits=2),
        A.RandomInvert(p=0.2),
        A.RandomSolarize(threshold=50.0),
        A.RandomAdjustSharpness(sharpness_factor=2),
        A.RandomAutocontrast(p=0.2),
        A.RandomEqualize(p=0.2),
        A.ColorJitter(0.5, 0.3),
        A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)), 
        A.ToTensor(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tr_data = DataSet(path=args.data_path, jaad_path=args.jaad_path, data_set='train', frame=True, vel=True, balance=False, transforms=transform, seg_map=args.seg, h3d=args.H3D, forcast=args.forcast)
    te_data = DataSet(path=args.data_path, jaad_path=args.jaad_path, data_set='test', frame=True, vel=True, balance=args.balance, bh='all', t23=args.balance, transforms=transform, seg_map=args.seg, h3d=args.H3D, forcast=args.forcast)
    val_data = DataSet(path=args.data_path, jaad_path=args.jaad_path, data_set='val', frame=True, vel=True, balance=False, transforms=transform, seg_map=args.seg, h3d=args.H3D, forcast=args.forcast)

    tr = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(te_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return tr, te, val


def main(args):
    seed_everything(args.seed)
    try:
        m_feat = args.logdir.split('/')[-2].split('-')[2]
    except IndexError:
        m_feat = 'N'
    args.frames = True if 'I' in m_feat else False
    args.velocity = True if 'V' in m_feat else False
    args.seg = True if 'S' in m_feat else False
    args.forecast = True if 'F' in m_feat else False
    args.time_crop = True if 'T' in m_feat else False
    args.H3D = False if args.logdir.split('/')[-2].split('-')[-1] == 'h2d' else True

    tr, te, val = data_loader(args)
    mymodel = LitPedGraph(args, len(tr))

    if not Path(args.logdir).is_dir():
        os.mkdir(args.logdir)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.logdir,
        filename='epoch_{epoch:02d}',   # 你可以改成其他格式
        save_top_k=-1,                  # -1 表示保存所有 epoch
        every_n_epochs=1,               # 每个 epoch 都保存
        save_weights_only=True         # 不设置为 True，这样保存完整 ckpt
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices='auto',
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed',
    )

    if args.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(mymodel, tr)
        new_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {new_lr}")
        mymodel.hparams.lr = new_lr

    trainer.fit(mymodel, tr, val)
    torch.save(mymodel.model.state_dict(), args.logdir + 'last.pth')
    trainer.test(mymodel, te, ckpt_path='best')
    torch.save(mymodel.model.state_dict(), args.logdir + 'best.pth')
    print('finish')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser("Pedestrian prediction crossing")
    parser.add_argument('--logdir', type=str, default="./data/jaad-23-IVSFT/", help="logger directory for tensorboard")
    parser.add_argument('--device', type=str, default=0, help="GPU")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate to train')
    parser.add_argument('--data_path', type=str, default='./data/JAAD', help='Path to the train and test data')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and test")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader")
    parser.add_argument('--frames', type=bool, default=False, help='Activate the use of raw frames')
    parser.add_argument('--velocity', type=bool, default=False, help='Activate the use of the OBD and GPS velocity')
    parser.add_argument('--seg', type=bool, default=False, help='Use the segmentation map')
    parser.add_argument('--forcast', type=bool, default=False, help='Use the human pose forecasting data')
    parser.add_argument('--time_crop', type=bool, default=False)
    parser.add_argument('--H3D', type=bool, default=True, help='Use 3D human keypoints')
    parser.add_argument('--jaad_path', type=str, default='./JAAD')
    parser.add_argument('--balance', type=bool, default=False, help='Balance or not the dataset')
    parser.add_argument('--bh', type=str, default='all', help='all or bh, if use all samples or only samples with behavior labels')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--auto_lr_find', action='store_true', help='Enable auto learning rate finder')
    args = parser.parse_args()
    main(args)
