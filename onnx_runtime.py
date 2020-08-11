# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import onnx
import onnxruntime

onnx_model = onnx.load("Corr_opset11.onnx")
onnx.checker.check_model(onnx_model)

onnx_model = onnx.load("onnx_models/Gen_opset11.onnx")
onnx.checker.check_model(onnx_model)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession("Corr_opset11.onnx")
opt = TestOptions().parse()
opt.name = "ade20k"
torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')


def preprocess_input(data):
    if opt.dataset_mode == 'celebahq':
        glasses = data['label'][:, 1::2, :, :].long()
        data['label'] = data['label'][:, ::2, :, :]
        glasses_ref = data['label_ref'][:, 1::2, :, :].long()
        data['label_ref'] = data['label_ref'][:, ::2, :, :]
        if use_gpu():
            glasses = glasses.cuda()
            glasses_ref = glasses_ref.cuda()
    elif opt.dataset_mode == 'celebahqedge':
        input_semantics = data['label'].clone().cuda().float()
        data['label'] = data['label'][:, :1, :, :]
        ref_semantics = data['label_ref'].clone().cuda().float()
        data['label_ref'] = data['label_ref'][:, :1, :, :]
    elif opt.dataset_mode == 'deepfashion':
        input_semantics = data['label'].clone().cuda().float()
        data['label'] = data['label'][:, :3, :, :]
        ref_semantics = data['label_ref'].clone().cuda().float()
        data['label_ref'] = data['label_ref'][:, :3, :, :]

    # move to GPU and change data types
    if opt.dataset_mode != 'deepfashion':
        data['label'] = data['label'].long()



    # create one-hot label map
    if opt.dataset_mode != 'celebahqedge' and opt.dataset_mode != 'deepfashion':
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = opt.label_nc + 1 if opt.contain_dontcare_label \
            else opt.label_nc
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        label_map = data['label_ref'].long()
        label_ref = torch.FloatTensor(bs, nc, h, w).zero_()
        ref_semantics = label_ref.scatter_(1, label_map, 1)

    if opt.dataset_mode == 'celebahq':
        assert input_semantics[:, -3:-2, :, :].sum().cpu().item() == 0
        input_semantics[:, -3:-2, :, :] = glasses
        assert ref_semantics[:, -3:-2, :, :].sum().cpu().item() == 0
        ref_semantics[:, -3:-2, :, :] = glasses_ref
    return data['label'], input_semantics, data['image'], data['ref'], data['ref'], data[
        'label_ref'], ref_semantics


# test
for i, data_i in enumerate(dataloader):
    print('{} / {}'.format(i, len(dataloader)))
    if i * opt.batchSize >= opt.how_many:
        break
    imgs_num = data_i['label'].shape[0]
    # data_i['stage1'] = torch.ones_like(data_i['stage1'])
    # compute ONNX Runtime output prediction
    input_label, input_semantics, real_image, ref, ref_image, ref_label, ref_semantics = preprocess_input(
        data_i, )
    #print(ort_session.get_inputs()[1].name)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(ref_image), ort_session.get_inputs()[1].name: to_numpy(input_semantics),
                  ort_session.get_inputs()[2].name: to_numpy(ref_semantics)}
    ort_outs = ort_session.run(None, ort_inputs)
    for i in ort_outs:
        print(i.shape)
    print(ort_outs[0])
    out = model(data_i, mode='inference')
    print(out.keys())
    print(out['warp_out'])
    print(out['warp_out'].shape)
    np.testing.assert_allclose(to_numpy(out['warp_out']), ort_outs[0], rtol=1e-03, atol=1e-05)
    
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    break
    '''
    if opt.save_per_img:
        root = save_root + '/test_per_img/'
        if not os.path.exists(root + opt.name):
            os.makedirs(root + opt.name)
        imgs = out['fake_image'].data.cpu()
        try:
            imgs = (imgs + 1) / 2
            for i in range(imgs.shape[0]):
                if opt.dataset_mode == 'deepfashion':
                    name = data_i['path'][i].split('Dataset/DeepFashion/')[-1].replace('/', '_')
                else:
                    name = os.path.basename(data_i['path'][i])
                vutils.save_image(imgs[i:i + 1], root + opt.name + '/' + name,
                                  nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)
    else:
        if not os.path.exists(save_root + '/test/' + opt.name):
            os.makedirs(save_root + '/test/' + opt.name)

        if opt.dataset_mode == 'deepfashion':
            label = data_i['label'][:, :3, :, :]
        elif opt.dataset_mode == 'celebahqedge':
            label = data_i['label'].expand(-1, 3, -1, -1).float()
        else:
            label = masktorgb(data_i['label'].cpu().numpy())
            label = torch.from_numpy(label).float() / 128 - 1

        imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            vutils.save_image(imgs, save_root + '/test/' + opt.name + '/' + str(i) + '.png',
                              nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)
    '''