
import argparse
from Dataset import MyCOCODataset
from network import BeastNet
import pickle
from torchvision.ops import complete_box_iou_loss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

class CompleteIOULoss(nn.Module):

    def __init__(self, reduction='none'):
        super(CompleteIOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        loss = complete_box_iou_loss(output, target, self.reduction)
        return loss

def train(net, num_epochs, batch_size):

    model = net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load train, test and inv. map dictionaries
    with open('train_data.pkl', 'rb') as f:
        train_dict = pickle.load(f)
    with open('inv_map.pkl', 'rb') as f:
        inv_map = pickle.load(f)
    print("train and inv. map loaded.")

    torch.autograd.set_detect_anomaly(True)

    # train dataset
    train_data = MyCOCODataset(train_dict, inv_map, train=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_size = len(train_loader)
    # train parameters
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = CompleteIOULoss('mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    # train history
    train_history = {"cross_loss": [],
                     "bbox_loss" : [],
                     "acc" : []}

    print("training is in progress...")
    for epoch in range(1, num_epochs+1):
        run_cross_loss = 0.0
        run_reg_loss = 0.0
        run_acc = 0.0
        for i, data in enumerate(train_loader):
            img, bbox, label = data['image'], data['bbox'], data['label']
            # load data to device
            img = img.to(device)
            bbox = bbox.to(device)
            label = label.to(device)

            # start train
            optimizer.zero_grad()
            output = model(img)
            pred_cat = output[0]
            pred_bbox = output[1]

            # calculate loss
            cross_loss = criterion1(pred_cat, label)
            cross_loss.backward(retain_graph=True)
            reg_loss = criterion2(pred_bbox, bbox)
            reg_loss.backward()
            optimizer.step()

            # accuracy
            _, pred = torch.max(pred_cat, 1)
            acc = torch.eq(pred, label).float().mean().item()

            run_cross_loss += cross_loss.item()
            run_reg_loss += reg_loss.item()
            run_acc += acc

        # calculate mean evaluations
        run_cross_loss /= loader_size
        run_reg_loss /= loader_size
        run_acc /= loader_size
        # report results
        print(f"[epoch {epoch}/{num_epochs}] train cross-entropy loss: {round(run_cross_loss,2)}")
        print(f"[epoch {epoch}/{num_epochs}] train regression loss: {round(run_reg_loss, 2)}")
        print(f"[epoch {epoch}/{num_epochs}] train classification accuracy {round(run_acc*100, 2)}")
        print("*"*30)
        # save the losses
        train_history["cross_loss"].append(run_cross_loss)
        train_history["bbox_loss"].append(run_reg_loss)
        train_history["acc"].append(run_acc)

    return {"model" : model,
            "train_history" : train_history}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=15, help="numbere of epochs for training")
    parser.add_argument("--batch_size", default=16, help="batch size")

    args = parser.parse_args()

    EPOCH = args.batch_size
    BATCH_SIZE = args.batch_size

    model = BeastNet()
    results = train(model, EPOCH, BATCH_SIZE)




