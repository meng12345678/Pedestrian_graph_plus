import torch
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from sklearn.metrics import balanced_accuracy_score

from jaad_dataloader23 import DataSet
from models.ped_graph23 import pedMondel

import os
import glob
import numpy as np
import argparse
from pathlib import Path


def seed_everything(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ModelTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置模型参数
        try:
            m_feat = args.logdir.split('/')[-2].split('-')[2]
        except IndexError:
            m_feat = 'N'
        self.frames = True if 'I' in m_feat else False
        self.velocity = True if 'V' in m_feat else False
        self.seg = True if 'S' in m_feat else False
        self.forecast = True if 'F' in m_feat else False
        self.H3D = False if args.logdir.split('/')[-2].split('-')[-1] == 'h2d' else True
        
        # 加载测试数据
        self.test_loader = self._prepare_test_data()
        
        # 测试样本权重
        te_nsamples = [1871, 3204, 13037]
        self.te_weight = torch.from_numpy(np.min(te_nsamples) / te_nsamples).float().to(self.device)
        
    def _prepare_test_data(self):
        transform = A.Compose([
            A.ToPILImage(),
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        te_data = DataSet(
            path=self.args.data_path, 
            jaad_path=self.args.jaad_path, 
            data_set='test', 
            frame=True, 
            vel=True, 
            balance=self.args.balance, 
            bh='all', 
            t23=self.args.balance, 
            transforms=transform, 
            seg_map=self.seg, 
            h3d=self.H3D, 
            forcast=self.forecast
        )
        
        return DataLoader(
            te_data, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
    
    def test_single_model(self, model_path):
        """测试单个模型文件"""
        # 创建模型
        model = pedMondel(
            self.frames, 
            self.velocity, 
            seg=self.seg, 
            h3d=self.H3D, 
            n_clss=3
        ).to(self.device)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                f = batch[2].to(self.device) if self.frames else None
                v = batch[3].to(self.device) if self.velocity else None
                
                logits = model(x, f, v)
                w = None if self.args.balance else self.te_weight
                
                # 计算损失
                y_onehot = torch.FloatTensor(y.shape[0], 3).to(y.device).zero_()
                y_onehot.scatter_(1, y.long(), 1)
                loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)
                # 将y中标签2（irrelevant）当作0（no crossing）
                y_binary = y.clone()
                y_binary[y_binary == 2] = 0
                
                # 计算准确率
                probs = torch.softmax(logits, dim=1)
                crossing_score = probs[:, 1]
                preds = (crossing_score > 0.5).long()
                preds[preds == 2] = 0  # 将irrelevant类视为no crossing
                acc = accuracy(preds.view(-1), y_binary.view(-1), task='binary')
                
                # 累计结果
                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_acc += acc.item() * batch_size
                total_samples += batch_size
                
                # 保存预测结果用于计算balanced accuracy
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_binary.cpu().numpy())
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_acc * 100.0, balanced_acc * 100.0
    
    def test_all_epochs(self):
        """测试所有epoch的模型"""
        epoch_models_dir = os.path.join(self.args.logdir, 'epoch_models')
        
        if not os.path.exists(epoch_models_dir):
            print(f"Error: Epoch models directory not found: {epoch_models_dir}")
            return
        
        # 获取所有epoch模型文件
        model_files = glob.glob(os.path.join(epoch_models_dir, 'epoch_*.pth'))
        model_files.sort()  # 按文件名排序
        
        if not model_files:
            print(f"No epoch model files found in {epoch_models_dir}")
            return
        
        print(f"Found {len(model_files)} epoch models to test")
        print("=" * 80)
        
        results = []
        best_acc = 0.0
        best_epoch = -1
        best_model_path = ""
        
        for model_file in model_files:
            # 从文件名提取epoch编号
            epoch_num = int(os.path.basename(model_file).split('_')[1].split('.')[0])
            
            print(f"Testing Epoch {epoch_num:02d}...")
            
            try:
                test_loss, test_acc, balanced_acc = self.test_single_model(model_file)
                
                results.append({
                    'epoch': epoch_num,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'balanced_acc': balanced_acc,
                    'model_path': model_file
                })
                
                print(f"Epoch {epoch_num:02d} - Test Loss: {test_loss:.4f}, "
                      f"Test Acc: {test_acc:.3f}%, Balanced Acc: {balanced_acc:.3f}%")
                
                # 更新最佳结果
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch_num
                    best_model_path = model_file
                    
            except Exception as e:
                print(f"Error testing epoch {epoch_num}: {str(e)}")
                continue
        
        print("=" * 80)
        print("TESTING SUMMARY:")
        print("=" * 80)
        
        # 排序并显示所有结果
        results.sort(key=lambda x: x['test_acc'], reverse=True)
        
        print(f"{'Rank':<4} {'Epoch':<6} {'Test Loss':<12} {'Test Acc (%)':<12} {'Balanced Acc (%)':<15}")
        print("-" * 60)
        
        for i, result in enumerate(results):
            print(f"{i+1:<4} {result['epoch']:<6} {result['test_loss']:<12.4f} "
                  f"{result['test_acc']:<12.3f} {result['balanced_acc']:<15.3f}")
        
        print("=" * 80)
        print(f"BEST MODEL:")
        print(f"Epoch: {best_epoch}")
        print(f"Test Accuracy: {best_acc:.3f}%")
        print(f"Model Path: {best_model_path}")
        print("=" * 80)
        
        # 保存结果到文件
        results_file = os.path.join(self.args.logdir, 'all_epochs_test_results.txt')
        with open(results_file, 'w') as f:
            f.write("Epoch Testing Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Epoch':<6} {'Test Loss':<12} {'Test Acc (%)':<12} {'Balanced Acc (%)':<15}\n")
            f.write("-" * 50 + "\n")
            
            for result in sorted(results, key=lambda x: x['epoch']):
                f.write(f"{result['epoch']:<6} {result['test_loss']:<12.4f} "
                       f"{result['test_acc']:<12.3f} {result['balanced_acc']:<15.3f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Best Model: Epoch {best_epoch}, Test Accuracy: {best_acc:.3f}%\n")
            f.write(f"Best Model Path: {best_model_path}\n")
        
        print(f"Results saved to: {results_file}")


def main(args):
    seed_everything(args.seed)
    
    tester = ModelTester(args)
    tester.test_all_epochs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test all epoch models")
    parser.add_argument('--logdir', type=str, default="./data/jaad-23-IVSFT-h2d/", 
                       help="logger directory where epoch models are saved")
    parser.add_argument('--data_path', type=str, default='./data/JAAD', 
                       help='Path to the train and test data')
    parser.add_argument('--jaad_path', type=str, default='./JAAD')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help="Batch size for testing")
    parser.add_argument('--num_workers', type=int, default=0, 
                       help="Number of workers for the dataloader")
    parser.add_argument('--balance', type=bool, default=True, 
                       help='Balance or not the dataset')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
