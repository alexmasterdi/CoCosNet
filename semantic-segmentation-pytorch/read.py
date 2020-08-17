import numpy as np
import sys
import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from argparse import ArgumentParser, SUPPRESS
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from scipy.io import loadmat
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument(
        "--image",
        required=True,
        metavar="FILE",
        help="path to image file",
        type=str,
    )
    return parser


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tr = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    img = tr(torch.from_numpy(img.copy()))
    print(img.shape)
    img = img.numpy()
    print(img.shape)
    return np.array([img])


def preprocess(image_path):
	img = Image.open(image_path).convert('RGB')
	w, h = img.size
	img = img.resize(size=(320, 320), resample=Image.BILINEAR)
	img = img_transform(img)
	return img


def seg_inference(core, path):
    path_seg = path
    args = build_argparser().parse_args()
    network = core.read_network(path_seg)
    input_blob = next(iter(network.input_info))
    output_blob = next(iter(network.outputs))
    exec_net = core.load_network(network=network, device_name="CPU")
    image = preprocess(args.image)
    print(image.shape)
    print(network.input_info[input_blob].input_data.shape, network.outputs[output_blob].shape)
    result = exec_net.infer(inputs={input_blob: image})
    return result[output_blob]


def decoder_inference(core, input):
    path_decoder = "onnx_models/decoder.onnx"
    network = core.read_network(path_decoder)
    input_blob = next(iter(network.input_info))
    output_blob = next(iter(network.outputs))
    print(network.input_info[input_blob].input_data.shape)
    print(network.outputs[output_blob].shape)
    exec_net = core.load_network(network=network, device_name="CPU")
    result = exec_net.infer(inputs={input_blob: input})
    return result[output_blob]


def visualize_result(pred):
    # print predictions in descending order
    pred = np.int32(pred)

    # colorize prediction
    colors = loadmat('data/color150.mat')['colors']
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    img_name = "res_seg.jpg"
    Image.fromarray(pred_color).save(
        os.path.join("./", img_name.replace('.jpg', '.png')))


def main():
    core = IECore()
    result = seg_inference(core)
    pred = np.argmax(result, axis=1)
    pred = np.squeeze(pred, 0)
    print(pred, pred.shape)
    visualize_result(pred)

if __name__ == '__main__':
    sys.exit(main() or 0)
