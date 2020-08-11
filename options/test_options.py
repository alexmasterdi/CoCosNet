"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--save_per_img', action='store_true', help='if specified, save per image')
        parser.add_argument('--show_corr', action='store_true', help='if specified, save bilinear upsample correspondence')
        parser.add_argument('--save_mode', type=str, default='openvino', help='Select one of these modes: source, openvino, onnx_runtime')
        parser.add_argument('--inference_mode', type=str, default='openvino', help='Select one of these modes: source, openvino, onnx_runtime')
        parser.add_argument('--gen_path', type=str, default="onnx_models/Gen_opset11.onnx", help="Input path to generate model")
        parser.add_argument('--corr_path', type=str, default="onnx_models/Corr_opset11.onnx", help="Input path to correspondence model")

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
