import os
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import copy
from bayes_opt import BayesianOptimization  # 导入贝叶斯优化库
import pandas as pd  # 用于保存 Excel 文件

# %% 数据预处理，读取数据并归一化

data_dir = r'C:\Users\TeFuriy\Desktop\PaperWork\data'
batch_size = 8
# 数据预处理
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载训练集并划分出验证集
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 加载测试集
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)

# 创建相应的数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# %%准确度计算函数evaluate_accuracy与训练函数train

def evaluate_accuracy(test_iter, net, loss, device):
    save_path = r"C:\Users\TeFuriy\Desktop\PaperWork\Answer\rate_test000.pth"
    best_model_test = net
    best_r_test = 0.0
    count = 0
    target_acc_sum = []
    target_number = []
    target_l_sum = []
    with torch.no_grad():
        for data, target in test_iter:
            data, target = Variable(data.float(), requires_grad=False), Variable(target)
            data = data.to(device)
            net.eval()
            target = target.to(device, dtype=torch.int64)
            target_hat = net(data)
            max = target_hat.argmax(dim=1)
            l = loss(target_hat, target)

            target_l_sum.append(l.cpu().item())

            max = target_hat.argmax(dim=1)
            target_acc_sum.append((max.eq(target).float()).sum().item())
            target_number.append(target.shape[0])

            count += 1

    print("test", sum(target_acc_sum), sum(target_number))
    test_acc = sum(target_acc_sum) / sum(target_number)
    test_loss = np.mean(target_l_sum)

    if test_acc > best_r_test:
        best_r_test = test_acc
        best_model_test = copy.deepcopy(net)
        torch.save(net.state_dict(), save_path)  # 保存权重文件

    return test_acc, test_loss


def train(train_iter, val_iter, net, loss, optimizer, device, num_epochs, excel_path, patience=5):
    best_r = 0.0
    best_model = None
    save_path = r"C:\Users\TeFuriy\Desktop\PaperWork\Answer\rate000.pth"
    net.train()
    loss_path = []
    train_acc_print = []
    val_acc_print = []
    val_loss_print = []
    net = net.to(device)
    print("Training on", device)

    # 早停机制变量
    epochs_no_improve = 0  # 记录验证集性能未提升的 epoch 数
    best_val_loss = float('inf')  # 初始化最佳验证集损失

    for epoch in range(num_epochs):
        train_l_sum = []
        train_acc_sum = []
        train_number = []
        start = time.time()

        for data, target in train_iter:
            data = data.to(device)
            target = target.to(device, dtype=torch.int64)
            data, target = data.clone().detach().requires_grad_(False), target.clone().detach()

            target_hat = net(data)
            l = loss(target_hat, target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum.append(l.cpu().item())
            max = target_hat.argmax(dim=1)
            train_acc_sum.append((max.eq(target).float()).sum().item())
            train_number.append(target.shape[0])

        train_acc = sum(train_acc_sum) / sum(train_number)
        train_loss = np.mean(train_l_sum)

        val_acc, val_loss = evaluate_accuracy(val_iter, net, loss, device)
        net.train()

        # 检查验证集性能是否提升
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r = train_acc
            best_model = copy.deepcopy(net)
            torch.save(net.state_dict(), save_path)
            epochs_no_improve = 0  # 重置计数器
        else:
            epochs_no_improve += 1  # 验证集性能未提升

        # 打印日志
        print(f'Epoch {epoch + 1}, Loss {train_loss:.4f}, Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}, Val Loss {val_loss:.3f}, Time {time.time() - start:.1f} sec')

        # 记录结果
        loss_path.append(train_loss)
        train_acc_print.append(train_acc)
        val_acc_print.append(val_acc)
        val_loss_print.append(val_loss)

        # 早停检查
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
            break

    # 保存训练结果到 Excel
    history = {
        'Epoch': list(range(1, len(loss_path) + 1)),
        'Train Loss': loss_path,
        'Train Acc': train_acc_print,
        'Val Loss': val_loss_print,
        'Val Acc': val_acc_print
    }

    if excel_path is not None:
        df = pd.DataFrame(history)
        df.to_excel(excel_path, index=False)

    return loss_path, train_acc_print, val_acc_print




class NetTransformer(nn.Module):
    def __init__(self, net, num_classes=5, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super(NetTransformer, self).__init__()
        # 使用resnet的卷积层，但去除全连接层
        self.net = nn.Sequential(*list(net.children())[:-2])

        # 添加线性层调整嵌入维度
        self.embedding_layer = nn.Linear(512, d_model)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.net(x)  # [batch_size, 512, 7, 7]
        x = x.flatten(2)  # [batch_size, 512, 49]
        x = x.permute(2, 0, 1)  # [49, batch_size, 512]

        x = self.embedding_layer(x)  # [49, batch_size, d_model]
        x = self.transformer_encoder(x)  # Transformer编码

        x = x.mean(dim=0)  # 对序列的所有元素取平均 [batch_size, d_model]

        x = self.fc(x)  # 分类头
        return x



# 定义目标函数
# 创建一个列表来存储优化过程
optimization_history = []


def train_evaluate(lr, weight_decay, dropout, d_model_multiplier, nhead, num_layers, dim_feedforward, num_epochs=30,
                   patience=5):
    # 计算实际的d_model值
    d_model = int(d_model_multiplier) * 8

    net = models.resnet18(pretrained=True)
    net_transformer = NetTransformer(
        net,
        d_model=d_model,
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        dropout=dropout
    )
    optimizer = optim.Adam(net_transformer.parameters(), lr=lr, weight_decay=weight_decay)

    # 训练和验证
    loss = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 记录当前迭代的参数
    current_params = {
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward
    }

    _, _, val_acc = train(train_loader, val_loader, net_transformer, loss, optimizer,
                          device, num_epochs, excel_path=None, patience=patience)

    # 记录本次迭代结果
    current_params['validation_accuracy'] = val_acc[-1]
    optimization_history.append(current_params)

    return val_acc[-1]


# 定义超参数范围
pbounds = {
    'lr': (0.00001, 0.00015),
    'weight_decay': (0.00001, 0.001),
    'dropout': (0.5, 0.7),
    'd_model_multiplier': (16, 64),  # 256-512范围，且一定是8的倍数
    'nhead': (8, 8),
    'num_layers': (1, 4),
    'dim_feedforward': (256, 4096)
}


# 贝叶斯优化
optimizer = BayesianOptimization(f=train_evaluate, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=20)

# 保存优化历史到Excel
optimization_df = pd.DataFrame(optimization_history)
optimization_df.to_excel(r'C:\Users\TeFuriy\Desktop\PaperWork\Answer\optimization_history_with_transformer.xlsx', index=False)



# 最佳超参数
best_params = optimizer.max['params']
print("Best Parameters:", best_params)

# 保存最佳超参数到 Excel
best_params_df = pd.DataFrame([best_params])
best_params_df.to_excel(r'C:\Users\TeFuriy\Desktop\PaperWork\Answer\best_params.xlsx', index=False)


# 使用优化后的超参数进行消融实验
def ablation_study_with_best_params(train_loader, val_loader, test_loader, device, best_params, num_epochs=100, patience=5):
    results = {}
    # 计算实际的d_model值
    d_model = int(best_params['d_model_multiplier']) * 8
    loss = nn.CrossEntropyLoss()

    # 1. 未训练的ResNet
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 5)
    optimizer = optim.Adam(net.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    excel_path = r'C:\Users\TeFuriy\Desktop\PaperWork\Answer\untrained_resnet_history.xlsx'
    l_p, train_a, val_a = train(train_loader, val_loader, net, loss, optimizer, device, num_epochs, excel_path, patience)
    test_acc, _ = evaluate_accuracy(test_loader, net, loss, device)
    results['untrained_resnet'] = {'train_acc': train_a, 'val_acc': val_a, 'test_acc': test_acc}

    # 2. 预训练的ResNet
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 5)
    optimizer = optim.Adam(net.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    excel_path = r'C:\Users\TeFuriy\Desktop\PaperWork\Answer\pretrained_resnet_history.xlsx'
    l_p, train_a, val_a = train(train_loader, val_loader, net, loss, optimizer, device, num_epochs, excel_path, patience)
    test_acc, _ = evaluate_accuracy(test_loader, net, loss, device)
    results['pretrained_resnet'] = {'train_acc': train_a, 'val_acc': val_a, 'test_acc': test_acc}

    # 3. 未训练的ResNet+Transformer
    net = models.resnet18(pretrained=False)
    net_transformer = NetTransformer(
        net,
        d_model=d_model,  # 使用计算好的d_model
        nhead=int(best_params['nhead']),
        num_layers=int(best_params['num_layers']),
        dim_feedforward=int(best_params['dim_feedforward']),
        dropout=best_params['dropout']
    )
    optimizer = optim.Adam(net_transformer.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    excel_path = r'C:\Users\TeFuriy\Desktop\PaperWork\Answer\untrained_resnet_transformer_history.xlsx'
    l_p, train_a, val_a = train(train_loader, val_loader, net_transformer, loss, optimizer, device, num_epochs,
                                excel_path, patience)
    test_acc, _ = evaluate_accuracy(test_loader, net_transformer, loss, device)
    results['untrained_resnet_transformer'] = {'train_acc': train_a, 'val_acc': val_a, 'test_acc': test_acc}

    # 4. 预训练的ResNet+Transformer
    net = models.resnet18(pretrained=True)
    net_transformer = NetTransformer(
        net,
        d_model=d_model,  # 使用计算好的d_model
        nhead=int(best_params['nhead']),
        num_layers=int(best_params['num_layers']),
        dim_feedforward=int(best_params['dim_feedforward']),
        dropout=best_params['dropout']
    )
    optimizer = optim.Adam(net_transformer.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    excel_path = r'C:\Users\TeFuriy\Desktop\PaperWork\Answer\pretrained_resnet_transformer_history.xlsx'
    l_p, train_a, val_a = train(train_loader, val_loader, net_transformer, loss, optimizer, device, num_epochs,
                                excel_path, patience)
    test_acc, _ = evaluate_accuracy(test_loader, net_transformer, loss, device)
    results['pretrained_resnet_transformer'] = {'train_acc': train_a, 'val_acc': val_a, 'test_acc': test_acc}

    return results

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用优化后的超参数进行消融实验
results = ablation_study_with_best_params(train_loader, val_loader, test_loader, device, best_params)

# 输出结果
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  Train Accuracy: {result['train_acc'][-1]:.3f}")
    print(f"  Validation Accuracy: {result['val_acc'][-1]:.3f}")
    print(f"  Test Accuracy: {result['test_acc']:.3f}")