import sys, os
sys.path.append('./Classify')
import torch 
from torch import nn
from mobilenet import MobileNetV2
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image 


def load_model(weight_pth, n_class=2):
    """
        weight_pth: str, 已经重训练好的权重
        n_class: int, 分类层的类数
    """
    weight_pth = os.path.expanduser(weight_pth)
    model = MobileNetV2()
    model.classifier = nn.Sequential(
        model.classifier[0],
        nn.Linear(in_features=1280, out_features=n_class)
    )
    model.load_state_dict(torch.load(weight_pth))
    return model


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


cps = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])


# IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def is_valide_file(entry):
    return entry.is_file() and (not entry.name.startswith('.')) 


class torchDataset(Dataset):
    """ A generic dataset where the images are arranged in this way:  png for example
        root/c1_001.png
        root/c1_002.png
        root/c1_003.png
        ...
        root/c2_001.png
        root/c2_002.png
    """
    def __init__(self, root, transforms=None, label_transforms=None):
        super(torchDataset, self).__init__()
        
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.label_transforms = label_transforms
        
        self.image_paths, self.labels = self._find_samples(self.root)
        self.classes = sorted(set(self.labels))
        self.class2idx = {v:i for i,v in enumerate(self.classes)}


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = pil_loader(self.image_paths[idx])
        label = self.class2idx[self.labels[idx]]
        if self.transforms:
            img = self.transforms(img)
        if self.label_transforms:
            label = self.label_transforms(label)
        return img, label

    def _find_samples(self, dir_):
        samples = [(d.path, d.name.split('_')[0]) for d in os.scandir(dir_) if is_valide_file(d)]
        images, labels = list(zip(*samples))
        return images, labels


def evaluation(model, ds, device, batch_size=128):
    model = model.to(device)

    dl = DataLoader(ds, batch_size=batch_size)
    ds_size = len(ds)

    corrects = 0
    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device)
        # print(images.shape, labels)
        # break
        with torch.no_grad():
            logits = model(images)
            _, preds = logits.max(dim=1)
            corrects += (preds==labels).sum()
        
    acc = corrects.double() / ds_size
    return acc


def load_dataset(ds_pth, cps, mode=1):
    """
        导入数据集，两种模式：
        mode 0：数据目录下每个类别的图片数据放在一个类名命名的文件夹中，如：
            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/xxz.png

            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/asd932_.png
        mode 1: 
    """
    assert mode in [0, 1], "mode must be 0(all store in a subdir separatly) or 1(all store in the rootdir)"
    if mode == 0:
        ds = datasets.ImageFolder(ds_pth, cps)
    else:
        ds = torchDataset(ds_pth, cps)
    return ds



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model('./Classify/models/best_model_manNhat_wt.pkl')
    model.eval()
    ds_pth = '~/Datasets/2_avi_01_vt4c/recall3/'
    ds = load_dataset(ds_pth, cps, mode=1)

    acc = evaluation(model, ds, device)
    print('recall rate: {:.4f}'.format(acc))


