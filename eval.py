import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import sys
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from argparse import ArgumentParser, SUPPRESS
import cv2
from PIL import Image
import logging as log
from openvino.inference_engine import IECore



torch.manual_seed(0)




def preprocess_input(data, opt):
# create one-hot label map
        data['label'] = data['label'].long()
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = opt.label_nc + 1 if opt.contain_dontcare_label \
            else opt.label_nc
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
    
        label_map = data['label_ref'].long()
        label_ref = torch.FloatTensor(bs, nc, h, w).zero_()
        ref_semantics = label_ref.scatter_(1, label_map, 1)

        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def save_results(save_root, data_i, result, options, index):
    if options.save_per_img:
        root = save_root + '/' + options.save_mode + '_per_img/'
        if not os.path.exists(root + options.name):
            os.makedirs(root + options.name)
        imgs = result.data.cpu()
        try:
            imgs = (imgs + 1) / 2
            for i in range(imgs.shape[0]):
                name = os.path.basename(data_i['path'][i])
                vutils.save_image(imgs[i:i+1], root + options.name + '/' + name,  
                        nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)
    else:
        imgs_num = data_i['label'].shape[0]
        if not os.path.exists(save_root + '/' + options.save_mode + '/' + options.name):
            os.makedirs(save_root + '/' + options.save_mode + '/' + options.name)
        label = masktorgb(data_i['label'].long().cpu().numpy())
        label = torch.from_numpy(label).float() / 128 - 1
    
        imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), result.data.cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            vutils.save_image(imgs, save_root + '/'+ options.save_mode + '/' + options.name + '/' + str(index) + '.png',  
                    nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)


class correspondence_model:
    def __init__(self, options):
        self.model_path = options.corr_path
        self.core = IECore()
        self.network = self.core.read_network(self.model_path)
        self.input_keys = list(self.network.input_info.keys())
        self.input_semantics = self.input_keys[0]
        self.reference_image = self.input_keys[1]
        self.reference_semantics = self.input_keys[2]
        self.output_keys = list(self.network.outputs.keys())
        self.warp_out = self.output_keys[0]
        self.warp_mask = self.output_keys[1]
        self.exec_net = self.core.load_network(network=self.network, device_name="CPU")


    def infer(self, input_sem, ref_image, ref_semantic):
        input = {}
        input[self.input_semantics] = to_numpy(input_sem)
        input[self.reference_image] = to_numpy(ref_image)
        input[self.reference_semantics] = to_numpy(ref_semantic)
        out = self.exec_net.infer(inputs=input)
        return out


class generate_model:
    def __init__(self, options):
        self.model_path = options.gen_path
        self.core = IECore()
        self.network = self.core.read_network(self.model_path)
        self.input_keys = list(self.network.input_info.keys())
        self.input_blob = self.input_keys[0]
        self.output_keys = list(self.network.outputs.keys())
        self.output_blob = self.output_keys[0]
        self.exec_net = self.core.load_network(network=self.network, device_name="CPU")


    def infer(self, input):
        inp = {}
        inp[self.input_blob] = to_numpy(input)
        out = self.exec_net.infer(inputs=inp)
        return out[self.output_blob]


def inference(options, data, model=None, Corr=None, Gen=None):
    out = []
    if options.inference_mode == 'pytorch':
        out = model(data, mode='inference')
        out = out['fake_image']
    if options.inference_mode == 'inference_engine':
        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = preprocess_input(data, options)
        corr_out = Corr.infer(input_semantics, ref_image, ref_semantics)
        gen_input = torch.cat((torch.from_numpy(corr_out[Corr.warp_out]), input_semantics), dim=1)
        out = torch.Tensor(Gen.infer(gen_input))
    return out


def init(options):
    models = {}
    models['model'] = None
    models['Corr'] = None
    models['Gen'] = None
    if options.inference_mode == 'pytorch':
        models['model'] = Pix2PixModel(options)
        models['model'].eval()
    else:
        models['Corr'] = correspondence_model(options)
        models['Gen'] = generate_model(options)
    return models


def main():
    # initialization
    opt = TestOptions().parse()
    opt.name = "ade20k"
    dataloader = data.create_dataloader(opt)
    dataloader.dataset[0]
    save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')
    models = init(opt)
    # inference
    for i, data_i in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)))
        if i * opt.batchSize >= opt.how_many:
            break
        result = inference(opt, data_i, models['model'], models['Corr'], models['Gen'])
        if opt.inference_mode == 'pytorch':
            opt.save_mode = 'source'
        if opt.inference_mode == 'openvino':
            opt.save_mode = 'openvino'
        print(data_i)
        save_results(save_root, data_i, result, opt, i)
        
        
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)

