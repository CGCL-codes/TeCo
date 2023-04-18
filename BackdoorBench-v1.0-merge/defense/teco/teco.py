'''
This file implements the (detection) defense method called TeCo (ft), which detects trigger samples during the inference stage based on corruption robustness consistency.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. TeCo detection:
        a. use image corruption
        b. get the hard-label output of backdoor-infected model (CRC test)
    4. use deviation for trigger sample detection
'''

import argparse
import logging
import os
import sys




sys.path.append('../')
sys.path.append(os.getcwd())
import torch
import numpy as np

from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.save_load_attack import load_attack_result
import yaml

from tqdm import tqdm
from PIL import Image
from imagecorruptions import corrupt
from sklearn import metrics
from sklearn.metrics import auc


def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')
    parser.add_argument('--yaml_path', type=str, default="./config/defense/teco/cifar10.yaml", help='the path of yaml')

    # dg settings
    parser.add_argument('--cor_type', type=str, help='type of image corruption')
    parser.add_argument('--severity', type=int, help='severity of image corruption')
    parser.add_argument('--max', type=int, default=6, help='max severity of image corruption')

    arg = parser.parse_args()

    print(arg)
    return arg


def dg(image, args):
    image = np.array(image)
    image = corrupt(image, corruption_name=args.cor_type, severity=args.severity)
    image = Image.fromarray(image)
    return image

def no_defense(args, result, config):
    model = generate_cls_model(args.model, args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    result = {}
    result['model'] = model
    return result

def save_defense_result_for_teco(
    clean_dict : dict,
    bd_dict : dict,
    save_path : str,
):

    save_dict = {
            'clean': clean_dict,
            'bd': bd_dict,
        }

    torch.save(
        save_dict,
        f'{save_path}/saved/teco/defense_result.pt',
    )


if __name__ == '__main__':
    ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "imagenet":
        args.num_classes = 1000
        args.input_height = 224
        args.input_width = 224
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    if args.model == 'swin_b4w7':
        args.input_height = 224
        args.input_width = 224
        args.batch_size = 16

    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/defence/teco/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save)
    if args.log is None:
        args.log = save_path + '/saved/teco/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)
    args.save_path = 'record/' + args.result_file

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')

    ### 3. no defense:
    result_defense = no_defense(args,result,config)

    ### 4. test the result and get ASR, ACC, RC
    result_defense['model'].eval()
    result_defense['model'].to(args.device)

    bd_dict = {}
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x, y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=False)

    for i, (inputs, labels) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        for j in range(len(pre_label)):
            save_name = str(i * args.batch_size + j)
            bd_dict[save_name] = {}
            bd_dict[save_name]['original'] = [pre_label[j].item()]
    for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                     'jpeg_compression']:
        for severity in range(1, 6):
            args.severity = severity
            args.cor_type = name
            x = result['bd_test']['x']
            for i in tqdm(range(len(x)), desc=f'{name} handling..., severity {severity}'):
                x[i] = dg(x[i], args)
            y = result['bd_test']['y']
            data_bd_test = list(zip(x,y))
            data_bd_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_bd_test,
                poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=False)

            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                for j in range(len(pre_label)):
                    save_name = str(i * args.batch_size + j)
                    if name not in bd_dict[save_name].keys():
                        bd_dict[save_name][name] = []
                        bd_dict[save_name][name].append(bd_dict[save_name]['original'][0])
                    bd_dict[save_name][name].append(pre_label[j].item())

    clean_dict = {}
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x, y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=False)

    for i, (inputs, labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        for j in range(len(pre_label)):
            save_name = str(i * args.batch_size + j)
            clean_dict[save_name] = {}
            clean_dict[save_name]['original'] = [pre_label[j].item()]
    for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                     'jpeg_compression']:
        for severity in range(1, 6):
            args.severity = severity
            args.cor_type = name
            x = result['clean_test']['x']
            for i in tqdm(range(len(x)), desc=f'{name} handling..., severity {severity}'):
                x[i] = dg(x[i], args)
            y = result['clean_test']['y']
            data_clean_test = list(zip(x,y))
            data_clean_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_clean_test,
                poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=False)

            for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                for j in range(len(pre_label)):
                    save_name = str(i * args.batch_size + j)
                    if name not in clean_dict[save_name].keys():
                        clean_dict[save_name][name] = []
                        clean_dict[save_name][name].append(clean_dict[save_name]['original'][0])
                    clean_dict[save_name][name].append(pre_label[j].item())

    result = {'clean': clean_dict, 'bd': bd_dict}
    save_defense_result_for_teco(
        clean_dict=clean_dict,
        bd_dict=bd_dict,
        save_path=args.save_path,
    )
    labels = []
    mads = []
    total_images = 0
    for file in ['clean', 'bd']:
        label_dict = result[file]
        images = list(label_dict.keys())
        keys = list(label_dict[images[0]].keys())
        total_images += len(images)
        for img in images:
            indexs = []
            img_preds = label_dict[img]
            for corruption in keys[1:]:
                flag = 0
                for i in range(args.max):
                    if int(img_preds[corruption][i]) != int(img_preds[corruption][0]):
                        index = i
                        flag = 1
                        indexs.append(index)
                        break
                if flag == 0:
                    indexs.append(args.max)
            indexs = np.asarray(indexs)
            mad = np.std(indexs)
            mads.append(mad)
            if file == 'clean':
                labels.append(0)
            else:
                labels.append(1)
    mads = np.asarray(mads)
    labels = np.asarray(labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, mads)
    f1_scores = []
    for th in thresholds:
        pred = np.where(mads > th, 1, 0)
        f1_score = metrics.f1_score(labels, pred, average='micro')
        f1_scores.append(f1_score)
    f1_scores = np.asarray(f1_scores)
    roc_auc = auc(fpr, tpr)

    defense_dict = {}
    defense_dict['fpr'] = fpr
    defense_dict['tpr'] = tpr
    defense_dict['thresholds'] = thresholds
    defense_dict['roc_auc'] = roc_auc
    defense_dict['f1_score'] = f1_scores
    result = defense_dict
    print(f"AUROC: {result['roc_auc']}")
    print(f"F1 SCORE: {np.max(result['f1_score'])}")
    print(f"saving...")
    torch.save(
        result,
        f'{args.save_path}/saved/teco/defense_result_roc.pt',
    )
    print(f"complete.")
