import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from src.utils import mixup, mixup_criterion
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
# from src.models.gcn import topk_loss
import copy
import math
from torch.autograd import Variable
from torch.autograd import Function
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    total_pred_list = []
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs
    best_train_acc, best_test_acc = 0, 0
    best_train_f1, best_test_f1 = 0, 0
    best_train_auc, best_test_auc = 0, 0

    for i in range(epoch_num):
        loss_all = 0
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()   #   criterion(torch.argmax(out, dim=1).float(), data.y.float())
        for data in train_loader:
            data = data.to(device)

            if args.mixup:
                data, y_a, y_b, lam = mixup(data)
            optimizer.zero_grad()
            out,loss_set,att_tsne= model(data)

            if args.mixup:
                out = out.long()
                loss = loss_set(data.y)
                #print(loss)
                # loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
                # loss = 0.5*mixup_criterion(F.nll_loss, out, y_a, y_b, lam)  + 0.5*topk_loss(s1, args.ratio)
            else:
                loss = loss_set(out, data.y)
                # loss = F.nll_loss(out, data.y)
            """if args.mixup:
                loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
                # loss = 0.5*mixup_criterion(F.nll_loss, out, y_a, y_b, lam)  + 0.5*topk_loss(s1, args.ratio)
            else:
                # loss = 0.5*F.nll_loss(out, data.y)+ 0.5*topk_loss(s1, args.ratio)"""
                # loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()

            loss_all += loss.item()
        epoch_loss = loss_all / len(train_loader.dataset)

        train_acc, train_auc, train_macro, best_train_acc, best_train_f1, best_train_auc = evaluate(model, device, best_train_acc, best_train_f1, best_train_auc,train_loader)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_acc={(train_acc * 100):.2f}, train_f1={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f},  '
                     f'best_train_acc={(best_train_acc * 100):.2f},'
                     f'best_train_f1={(best_train_f1 * 100):.2f},'
                     f'best_train_auc={(best_train_auc * 100):.2f}')

        if (i + 1) % args.test_interval == 0:
            test_acc, test_auc, test_macro, best_acc, best_f1, best_auc = evaluate(model, device, best_test_acc, best_test_f1, best_test_auc, test_loader, test_loader)
            accs.append(test_acc)
            aucs.append(test_auc)
            macros.append(test_macro)
            text = f'(Train Epoch {i}), test_acc={(test_acc * 100):.2f}, ' \
                   f'test_f1={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f},' \
                   f'best_test_acc={(best_acc * 100):.2f},' \
                   f'best_test_f1={(best_f1 * 100):.2f}, best_test_auc={(best_auc * 100):.2f} ' \
                   f' \n'
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    return accs.mean(), aucs.mean(), macros.mean()


@torch.no_grad()
def evaluate(model, device, best_acc, best_f1, best_auc, loader, test_loader: Optional[DataLoader] = None, vis_flag: bool = False) -> (float, float):
    model.eval()
    preds, trues, preds_prob = [], [], []
    tsne_list = []

    correct, auc = 0, 0
    for data in loader:
        data = data.to(device)
        c,loss_set,test_tsne_matrix= model(data)

        if vis_flag:
            tsne_list.append(test_tsne_matrix)
            tsne_app = torch.cat(tsne_list, dim=0)
            #print(tsne_app.shape)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()

        trues += data.y.detach().cpu().tolist()


    #print('preds')
    #print(len(preds))
    #print('preds_prob')
    #print(len(preds_prob))
    #print('trues')
    #print(len(trues))
    #print(len(total_pred_list))

    #\begin{tsne}
    if vis_flag:
        tsne_cpu=tsne_app.cpu()
        array = np.asarray(tsne_cpu)
        tsne = TSNE(n_components=2, perplexity=32, learning_rate=200)
        data_embed_2d = tsne.fit_transform(array)

        # visualization
        plt.figure(figsize=(10, 8))
        colors = ['goldenrod' if label == 0 else 'c' for label in preds]
        plt.scatter(data_embed_2d[:, 0], data_embed_2d[:, 1], c=colors)
        plt.title('t-SNE Visualization of Image Classification')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
    #\end{tsne}

    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(auc):
        train_auc = 0.5
    train_acc = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
    if train_acc > best_acc:
        best_acc = train_acc

    if train_macro > best_f1:
        best_f1 = train_macro

    if train_auc > best_auc:
        best_auc = train_auc


    if test_loader is not None:

        # train_acc, train_auc, train_macro, test_acc, test_auc, test_macro, best_acc, best_f1, best_auc = evaluate(model, device, best_acc, best_f1, best_auc, test_loader,test_loader)
        return train_acc, train_auc, train_macro, best_acc, best_f1, best_auc
    else:
        return train_acc, train_auc, train_macro, best_acc, best_f1, best_auc


######################################################### test Grad CAM only，evaluate2 in example_main.py ####################

@torch.no_grad()
def evaluate2(model, device, best_acc, best_f1, best_auc, loader, test_loader: Optional[DataLoader] = None) -> (float, float):
    model.eval()
    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
    for data in loader:
        #gcam = GradCam(model=model.eval(), use_cuda=True)

        #print(type(gcam))
        #print(gcam)

        data = data.to(device)
        features1, gradients1,features2, gradients2 = 0,0,0,0
        #features1, gradients1 = get_feature_gradients(data, model.requires_grad_(True), 0, gradients1, features1)
        #features2, gradients2 = get_feature_gradients(data, model.requires_grad_(True), 1, gradients2, features2)

        c,loss_set= model(data)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()

        trues += data.y.detach().cpu().tolist()

    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(auc):
        train_auc = 0.5
    train_acc = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
    if train_acc > best_acc:
        best_acc = train_acc

    if train_macro > best_f1:
        best_f1 = train_macro

    if train_auc > best_auc:
        best_auc = train_auc

    return train_acc, train_auc, train_macro, best_acc, best_f1, best_auc


def get_feature_gradients(data, model, target_class,gradients,features):
    # grad=0

    # forward propagation
    logits,loss_set = model(data)

    # obtain the score of the target class
    class_score = logits[:, target_class]

    model.zero_grad()

    # backward propagation for the target class
    class_score.backward()

    # Obtain the gradient of the last convolutional layer
    gradients += model.conv2.weight.grad.clone().detach()

    # Obtain the feature map of the last convolutional layer
    features += model.conv2.activation.detach()

    return features, gradients