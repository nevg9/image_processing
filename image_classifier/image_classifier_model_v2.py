from tqdm import tqdm
from torch.utils.data import random_split
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 图片预处理


class ImageTransform():

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                # data augmentation
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # convert to tensor for PyTorch
                transforms.ToTensor(),
                # color normalization
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):

        return self.data_transform[phase](img)


def read_file(file, pos=True):
    file_list = []
    for line in open(file):
        row = line.strip()
        if row == "":
            continue
        file_list.append(row)

    return file_list

# 把传进来的file list切分为训练集和验证集


def split_train_and_val(path_file, train_ratio, pos=True):
    file_list = read_file(path_file, pos)
    num_samples = len(file_list)
    train_size = int(train_ratio * num_samples)
    val_size = num_samples - train_size
    train_file_list, val_file_list = random_split(
        file_list, [train_size, val_size])
    print(f"训练集大小:{len(train_file_list)},验证集大小:{len(val_file_list)}")
    return train_file_list, val_file_list

# 训练数据DataSet


class forestDataset(data.Dataset):

    def __init__(self, pos_file_list, neg_file_list, transform=None, phase='train'):
        self.pos_file_list = pos_file_list
        self.neg_file_list = neg_file_list
        self.total_file_list = self.pos_file_list + self.neg_file_list
        self.transform = transform
        self.phase = phase

    # 判断图片路径是否有效
    def check_image_invalid(self, file_list, pos=True):
        valid_image_list = []
        if pos:
            fd = open('invald_image_pos_data.txt', 'w')
        else:
            fd = open('invald_image_neg_data.txt', 'w')

        for file in file_list:
            try:
                Image.open(file)
                valid_image_list.append(file)
            except Exception:
                fd.write(file + "\n")

        print(
            f"invalid image size:{len(file_list) - len(valid_image_list)},input image size:{len(file_list)}")
        return valid_image_list

    def __len__(self):
        return len(self.total_file_list)

    def __getitem__(self, index):

        # load image
        img_path = self.total_file_list[index]

        # 读取到的图片有问题，则返回None
        img_originalsize = Image.open(img_path)

        # resize
        img = img_originalsize.resize((256, 256))

        # preprocess
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        # picking up labels
        label = int(img_path in self.pos_file_list)

        return img_transformed, label

# training function


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    accuracy_list = []
    loss_list = []

    # Precondition : Accelerator GPU -> 'On'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device：", device)

    # put network into GPU
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # epoch loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # set network 'train' mode
            else:
                net.eval()   # set network 'val' mode

            epoch_loss = 0.0
            epoch_corrects = 0

            # Before training
            if (epoch == 0) and (phase == 'train'):
                continue

            # batch loop
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # send data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # initialize optimizer
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)  # calcurate loss
                    _, preds = torch.max(outputs, 1)  # predict

                    # back propagtion
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # update loss summation
                    epoch_loss += loss.item() * inputs.size(0)
                    # update correct prediction summation
                    epoch_corrects += torch.sum(preds == labels.data)

            # loss and accuracy for each epoch loop
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                accuracy_list.append(epoch_acc.item())
                loss_list.append(epoch_loss)

    return accuracy_list, loss_list


def get_pretrain_model():
    use_pretrained = True
    net = models.vgg16_bn(pretrained=use_pretrained)

    # Replace output layer for 2 class classifier,
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.train()
    return net


size = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_ratio = 0.9

train_pos_file_list, val_pos_file_list = split_train_and_val(
    path_file="./have_target_images.txt", train_ratio=train_ratio, pos=True)
train_neg_file_list, val_neg_file_list = split_train_and_val(
    path_file="./no_target_images.txt", train_ratio=train_ratio, pos=False)

train_dataset = forestDataset(train_pos_file_list, train_neg_file_list,
                              transform=ImageTransform(size, mean, std), phase='train')
val_dataset = forestDataset(val_pos_file_list, val_neg_file_list,
                            transform=ImageTransform(size, mean, std), phase='val')

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=28, prefetch_factor=10, drop_last=False)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=28, prefetch_factor=10, drop_last=False)

# put dataloader into dictionary type
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 使用vgg16预训练模型
# load pretrained vgg16 from PyTorch as an instance
# need to make setting 'internet' to 'On'.
net = get_pretrain_model()
# setting of loss function
criterion = nn.CrossEntropyLoss()

# setting fine tuned parameters

params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# Not only output layer, "features" layers and other classifier layers are tuned.
update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

# store parameters in list
for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        # print("params_to_update_1:", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        # print("params_to_update_2:", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        # print("params_to_update_3:", name)

    else:
        param.requires_grad = False
        # print("no learning", name)

# print("-----------")
# print(params_to_update_1)

# Learning Rates
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)

num_epochs = 10
accuracy_list, loss_list = train_model(
    net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

torch.save(net.state_dict, "image_classifier_model_vgg16bn_v2.pth")
