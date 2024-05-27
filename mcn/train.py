import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
import numpy as np  # Ensure numpy is imported
from sklearn import metrics

from model import CompatModel
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders

def train(model, device, train_loader, val_loader, comment):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    saver = BestSaver(comment)
    epochs = 50
    
    for epoch in range(1, epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))
        model.train()
        total_losses = AverageMeter()
        clf_losses = AverageMeter()
        vse_losses = AverageMeter()

        for batch_num, (lengths, images, names, offsets, set_ids, labels, is_compat) in enumerate(train_loader, 1):
            images = images.to(device)
            target = is_compat.float().to(device)
            
            optimizer.zero_grad()
            output, vse_loss, tmasks_loss, features_loss = model(images, names)
            output = output.squeeze(1)
            clf_loss = criterion(output, target)
            features_loss = 5e-3 * features_loss
            tmasks_loss = 5e-4 * tmasks_loss
            total_loss = clf_loss + vse_loss + features_loss + tmasks_loss
            
            total_losses.update(total_loss.item(), images.size(0))
            clf_losses.update(clf_loss.item(), images.size(0))
            vse_losses.update(vse_loss.item(), images.size(0))
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        logging.info("Epoch {} complete. Training loss: {:.4f}".format(epoch, total_losses.avg))
        evaluate(model, device, val_loader, criterion, saver, epoch)

def evaluate(model, device, val_loader, criterion, saver, epoch):
    model.eval()
    val_losses = AverageMeter()
    outputs = []
    targets = []

    for lengths, images, names, offsets, set_ids, labels, is_compat in val_loader:
        images = images.to(device)
        target = is_compat.float().to(device)
        
        with torch.no_grad():
            output, _, _, _ = model(images, names)
            output = output.squeeze(1)
            loss = criterion(output, target)
        
        val_losses.update(loss.item(), images.size(0))
        outputs.append(output)
        targets.append(target)

    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    accuracy = np.mean((outputs > 0.5) == targets)
    
    logging.info("Validation - Epoch: {} AUC: {:.4f}, Accuracy: {:.4f}, Loss: {:.4f}".format(epoch, auc, accuracy, val_losses.avg))
    saver.save(auc, model.state_dict())

def main():
    parser = argparse.ArgumentParser(description='Fashion Compatibility Training.')
    parser.add_argument('--vse_off', action="store_true")
    parser.add_argument('--pe_off', action="store_true")
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--conv_feats', type=str, default="1234")
    parser.add_argument('--comment', type=str, default="")
    args = parser.parse_args()

    config_logging(args.comment)
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders(batch_size=8, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                        vse_off=args.vse_off, pe_off=args.pe_off, mlp_layers=args.mlp_layers, conv_feats=args.conv_feats)
    
    train(model, device, train_loader, val_loader, args.comment)

if __name__ == '__main__':
    main()
