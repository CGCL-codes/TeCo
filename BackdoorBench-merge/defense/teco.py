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
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import yaml
import logging
import time
from defense.base import defense


from utils.aggregate_block.fix_random import fix_random
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_dataset_normalization
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
import torchvision.transforms as transforms
import copy
from copy import deepcopy
from imagecorruptions import corrupt
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import auc

def dg(image, name, severity):
    image = np.array(image)
    image = corrupt(image, corruption_name=name, severity=severity)
    image = Image.fromarray(image)
    return image


class corruptions(torch.nn.Module):
    def __init__(self, name, severity):
        super().__init__()
        self.name = name
        self.severity = severity

    def forward(self, img):
        return dg(img, self.name, self.severity)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, severity={self.severity})"

def get_transform_for_teco(dataset_name, input_height, input_width, train=True, random_crop_padding=4):
    # idea : given name, return the final implememnt transforms for the dataset
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=random_crop_padding))
        # transforms_list.append(transforms.RandomRotation(10))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms_list

def summary_dict(input_dict):
    '''
    Input a dict, this func will do summary for it.
    deepcopy to make sure no influence for summary
    :return:
    '''
    input_dict = deepcopy(input_dict)
    summary_dict_return = dict()
    for k,v in input_dict.items():
        if isinstance(v, dict):
            summary_dict_return[k] = summary_dict(v)
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            summary_dict_return[k] = {
                'shape':v.shape,
                'min':v.min(),
                'max':v.max(),
            }
        elif isinstance(v, list):
            summary_dict_return[k] = {
                'len':v.__len__(),
                'first ten':v[:10],
                'last ten':v[-10:],
            }
        else:
            summary_dict_return[k] = v
    return  summary_dict_return


class Args:
    pass


def load_attack_result_for_teco(
    save_path : str,
):
    '''
    This function first replicate the basic steps of generate models and clean train and test datasets
    then use the index given in files to replace the samples should be poisoned to re-create the backdoor train and test dataset

    save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path!!!
    save_path : the path of "attack_result.pt"
    '''
    load_file = torch.load(save_path)

    if all(key in load_file for key in ['model_name',
        'num_classes',
        'model',
        'data_path',
        'img_size',
        'clean_data',
        'bd_train',
        'bd_test',
        ]):

        logging.info('key match for attack_result, processing...')

        # model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
        # model.load_state_dict(load_file['model'])

        clean_setting = Args()

        clean_setting.dataset = load_file['clean_data']

        # convert the relative/abs path in attack result to abs path for defense
        clean_setting.dataset_path = load_file['data_path']
        logging.warning("save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path")
        clean_setting.dataset_path = save_path[:save_path.index('record')] + clean_setting.dataset_path[clean_setting.dataset_path.index('data'):]

        clean_setting.img_size = load_file['img_size']

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform = dataset_and_transform_generate(clean_setting)

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset_without_transform,
            train_img_transform,
            train_label_transform,
        )

        clean_test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset_without_transform,
            test_img_transform,
            test_label_transform,
        )

        if load_file['bd_train'] is not None:
            bd_train_dataset = prepro_cls_DatasetBD_v2(train_dataset_without_transform)
            bd_train_dataset.set_state(
                load_file['bd_train']
            )
            bd_train_dataset_with_transform = dataset_wrapper_with_transform(
                bd_train_dataset,
                train_img_transform,
                train_label_transform,
            )
        else:
            logging.info("No bd_train info found.")
            bd_train_dataset_with_transform = None


        bd_test_dataset = prepro_cls_DatasetBD_v2(test_dataset_without_transform)
        bd_test_dataset.set_state(
            load_file['bd_test']
        )
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        new_dict = copy.deepcopy(load_file['model'])
        for k, v in load_file['model'].items():
            if k.startswith('module.'):
                del new_dict[k]
                new_dict[k[7:]] = v

        test_img_transform_for_teco = get_transform_for_teco(clean_setting.dataset,
                                                             *(clean_setting.img_size[:2]),
                                                             train=False)
        load_file['model'] = new_dict
        load_dict = {
                'model_name': load_file['model_name'],
                'model': load_file['model'],
                'clean_train': clean_train_dataset_with_transform,
                'clean_test': clean_test_dataset_with_transform,
                'bd_train': bd_train_dataset_with_transform,
                'bd_test': bd_test_dataset_with_transform,
                'bd_test_img': bd_test_dataset,
                'clean_test_img': test_dataset_without_transform,
                'test_img_transform': test_img_transform_for_teco,
                'test_label_transform': test_label_transform,
            }

        print(f"loading...")

        return load_dict

    else:
        logging.info(f"loading...")
        logging.debug(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file

def save_defense_result_for_teco(
    clean_dict : dict,
    bd_dict : dict,
    save_path : str,
):

    save_dict = {
            'clean': clean_dict,
            'bd': bd_dict,
        }

    logging.info(f"saving...")
    logging.debug(f"location : {save_path}/defense_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{save_path}/defense_result.pt',
    )

class teco(defense):

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])

        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--model', type=str, help='resnet18')


        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/teco/config.yaml", help='the path of yaml')



    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/teco/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))
        self.args.save_path = save_path
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result_for_teco(attack_file + '/attack_result.pt')


    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device

    def CRC(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)

        test_transform = self.result['test_img_transform']
        test_label_transform = self.result['test_label_transform']
        data_bd_testset = self.result['bd_test_img']
        data_clean_testset = self.result['clean_test_img']

        logging.info(f"backdoor images clean testing...")
        bd_dict = {}
        bd_test_dataset_with_transform = self.result['bd_test']
        data_bd_loader = torch.utils.data.DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, drop_last=False, shuffle=False,
                                                     pin_memory=False)
        for i, items in enumerate(tqdm(data_bd_loader)):  # type: ignore
            inputs = items[0].to(args.device)
            outputs = model(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            for j in range(len(pre_label)):
                save_name = str(i * args.batch_size + j)
                bd_dict[save_name] = {}
                bd_dict[save_name]['original'] = [pre_label[j].item()]
        for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                     'jpeg_compression']:
            for severity in range(1, 6):
                logging.info(f"backdoor images corruption: {name}, severity: {severity} testing...")
                test_img_transform = test_transform
                test_img_transform.insert(0, corruptions(name, severity))
                test_img_transform = transforms.Compose(test_img_transform)
                bd_test_dataset_with_transform = dataset_wrapper_with_transform(
                    data_bd_testset,
                    test_img_transform,
                    test_label_transform,
                )
                data_bd_loader = torch.utils.data.DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size,
                                                             num_workers=args.num_workers, drop_last=False,
                                                             shuffle=False, pin_memory=False)
                for i, items in enumerate(tqdm(data_bd_loader)):  # type: ignore
                    inputs = items[0].to(args.device)
                    outputs = model(inputs)
                    pre_label = torch.max(outputs, dim=1)[1]
                    for j in range(len(pre_label)):
                        save_name = str(i * args.batch_size + j)
                        if name not in bd_dict[save_name].keys():
                            bd_dict[save_name][name] = []
                            bd_dict[save_name][name].append(bd_dict[save_name]['original'][0])
                        bd_dict[save_name][name].append(pre_label[j].item())

        logging.info(f"backdoor images complete.")

        logging.info(f"clean images clean testing...")
        clean_dict = {}
        clean_test_dataset_with_transform = self.result['clean_test']
        data_clean_loader = torch.utils.data.DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, drop_last=False, shuffle=False,
                                                     pin_memory=False)
        for i, items in enumerate(tqdm(data_clean_loader)):  # type: ignore
            inputs = items[0].to(args.device)
            outputs = model(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            for j in range(len(pre_label)):
                save_name = str(i * args.batch_size + j)
                clean_dict[save_name] = {}
                clean_dict[save_name]['original'] = [pre_label[j].item()]
        for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                     'jpeg_compression']:
            for severity in range(1, 6):
                logging.info(f"clean images corruption: {name}, severity: {severity} testing...")
                test_img_transform = test_transform
                test_img_transform.insert(0, corruptions(name, severity))
                test_img_transform = transforms.Compose(test_img_transform)
                clean_test_dataset_with_transform = dataset_wrapper_with_transform(
                    data_clean_testset,
                    test_img_transform,
                    test_label_transform,
                )
                data_clean_loader = torch.utils.data.DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size,
                                                             num_workers=args.num_workers, drop_last=False,
                                                             shuffle=False, pin_memory=False)
                for i, items in enumerate(tqdm(data_clean_loader)):  # type: ignore
                    inputs = items[0].to(args.device)
                    outputs = model(inputs)
                    pre_label = torch.max(outputs, dim=1)[1]
                    for j in range(len(pre_label)):
                        save_name = str(i * args.batch_size + j)
                        if name not in clean_dict[save_name].keys():
                            clean_dict[save_name][name] = []
                            clean_dict[save_name][name].append(clean_dict[save_name]['original'][0])
                        clean_dict[save_name][name].append(pre_label[j].item())

        logging.info(f"clean images complete.")

        result = {'clean': clean_dict, 'bd': bd_dict}
        save_defense_result_for_teco(
            clean_dict=clean_dict,
            bd_dict=bd_dict,
            save_path=args.save_path,
        )
        return result

    def detection(self, result_file, max=6):
        self.set_result(result_file)
        self.set_logger()
        result = self.CRC()

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
                    for i in range(max):
                        if int(img_preds[corruption][i]) != int(img_preds[corruption][0]):
                            index = i
                            flag = 1
                            indexs.append(index)
                            break
                    if flag == 0:
                        indexs.append(max)
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
        logging.info(f"AUROC: {result['roc_auc']}")
        logging.info(f"F1 SCORE: {np.max(result['f1_score'])}")
        logging.info(f"saving...")
        torch.save(
            result,
            f'{args.save_path}/defense_result_roc.pt',
        )
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    teco.add_arguments(parser)
    args = parser.parse_args()
    teco_method = teco(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = teco_method.detection(args.result_file)