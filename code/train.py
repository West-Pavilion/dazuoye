import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     # transforms.CenterCrop(224),
                                     # transforms.RandomResizedCrop(224),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = f"/home/tc/tucao201/zqz/diseases_pests"  # TODO flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 128     # TODO
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net = resnet34()
    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet34-b627a593.pth"  # TODO
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # # for param in net.parameters():
    # #     param.requires_grad = False
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    # net = models.resnet34(pretrained=True)

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)  # TODO
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)  # TODO

    epochs = 10  # TODO
    valid_step_interval = 10  # TODO
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    writer = SummaryWriter("./logs")
    for epoch in range(epochs):
        # train
        # net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            net.train()
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            accuracy = torch.eq(torch.max(logits, dim=1)[1], labels.to(device)).sum().item() / batch_size
            writer.add_scalar("train_loss", loss, epoch * train_steps + step + 1)
            writer.add_scalar("train_accuracy", accuracy, epoch * train_steps + step + 1)

            # print statistics
            running_loss += loss.item()     # 暂时无用

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} accuracy:{:.3f}".format(epoch + 1, epochs, loss, accuracy)

            if (epoch * train_steps + step) % valid_step_interval == 0:  # TODO
                # validate
                net.eval()
                val_loss = 0.0
                acc = 0.0  # accumulate accurate number
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = net(val_images.to(device))
                        loss = loss_function(outputs, val_labels.to(device))
                        val_loss += loss.item()
                        predict_y = torch.max(outputs, dim=1)[1]

                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                        val_bar.desc = "valid epoch[{}/{}], step: {}, accuracy: {:.3f}".format(epoch + 1, epochs,
                                                                                               epoch * train_steps + step + 1,
                                                                                               acc / val_num)

                val_accuracy = acc / val_num

                writer.add_scalar("val_loss", val_loss / val_steps,
                                  (epoch * train_steps + step) // valid_step_interval + 1)
                writer.add_scalar("val_accuracy", val_accuracy, (epoch * train_steps + step) // valid_step_interval + 1)

                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    torch.save(net.state_dict(), save_path)

    writer.close()

    # 计算训练集准确率
    model = models.resnet34()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 4)  # TODO
    model.to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    acc = 0.0  # accumulate accurate number
    with torch.no_grad():
        val_bar = tqdm(train_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "train accuracy: {:.3f}".format(acc / train_num)

    train_accuracy = acc / train_num

    print(f'best_acc: {best_acc}')
    print(f'train_acc: {train_accuracy}')
    print('Finished Training')


if __name__ == '__main__':
    main()

# best_acc: 0.9990727002967359
# train_acc: 0.9999536350148368
