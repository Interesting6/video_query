import sys, os
sys.path.append('./Classify')
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import numpy as np
import time, copy
from mobilenet import MobileNetV2


def load_model(weight_pth, n_class=2, feature_extracting=True):
    """
        weight_pth: str, 官方预训练好的mobilenet的权重路径
        n_class: int, 分类层的类数
        feature_extracting, bool, 默认为True表示只训练最后一层，若为False表示全局微调
    """

    model = MobileNetV2()
    model.load_state_dict(torch.load(weight_pth))

    if feature_extracting:
        for param in model.parameters(): 
            param.requires_grad = False

    logits_in = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=logits_in, out_features=n_class)
    return model


# print(model)



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_dls(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class2idx = image_datasets['train'].class_to_idx
    print(class2idx)
    return image_datasets, dataloaders, dataset_sizes




def train_model(model, dataloaders, criterion, optimizer, scheduler, device, wt_save_pth, num_epochs=30):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    torch.save(best_model_wts, wt_save_pth)
    print('best model weights saved!')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_class = 2
    lr = 0.001
    step_size = 7

    weight_pth = '/root/codes/RemoteWork/Classify/models/mobilenet_v2_wt.pkl'
    model = load_model(weight_pth)


    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    data_dir = '/root/Datasets/photos'
    image_datasets, dataloaders, dataset_sizes = load_dls(data_dir)

    wt_save_pth = '/root/codes/RemoteWork/Classify/models/best_model_manNhat_wt.pkl'
    model, val_acc_history = train_model(model, dataloaders, criterion, 
                optimizer, scheduler, device, wt_save_pth, num_epochs=30)


