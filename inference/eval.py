from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-g", "--generator_network", help="Required. Path to an .onnx file with a trained generator model.", required=True,
                      type=str, default="../onnx_models/Gen_opset11.onnx")
    args.add_argument("-c", "--correspondence_network", help="Required. Path to an .onnx file with a trained correspondence model.", required=True,
                      type=str, default="../onnx_models/Corr_opset11.onnx")
    return parser


def main():
    args = build_argparser().parse_args()
    gen = args.generator_network
    corr = args.correspondence_network
    ie = IECore()
    net_g = ie.read_network(corr)
    for key in net_g.input_info.keys():
        print(net_g.input_info[key].input_data.shape)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)

