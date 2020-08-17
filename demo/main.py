import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
import design
import numpy as np
from PIL import Image
import sample as lib
import cv2
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-s", "--segmentation_network", help="Required. Path to the segmentation network",
    	              default="../onnx_models/Seg_opset9.onnx", type=str)
    args.add_argument("-c", "--correspondence_network", help="Required. Path to the correspondence_network",
    	              default="../onnx_models/Corr_opset11.onnx", type=str)
    args.add_argument("-g", "--generate_network", help="Required. Path to the generator",
    	              default="../onnx_models/Gen_opset11.onnx", type=str)
    return parser


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self, args):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.get_image)
        self.pushButton_2.clicked.connect(self.get_reference)
        self.pushButton_3.clicked.connect(self.inference)
        self.core = IECore()
        self.Corr = lib.correspondence_model(args.correspondence_network, self.core)
        self.Gen = lib.generate_model(args.generate_network, self.core)
        self.Seg = lib.segmentation_model(args.segmentation_network, self.core)
    

    def array2QImage(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        image = np.require(image, np.uint8, 'C')
        qImg = QImage(image, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg


    def get_image(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image File")

        if file_name:
            self.image_path = file_name[0]
            image = cv2.imread(file_name[0])
            image = cv2.resize(image, dsize=(300, 300), interpolation=Image.BICUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(self.array2QImage(image))
            self.label.setPixmap(pixmap)


    def get_reference(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open Reference File")

        if file_name: 
            self.ref_path = file_name[0]
            image = cv2.imread(file_name[0])
            image = cv2.resize(image, dsize=(300, 300), interpolation=Image.BICUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(self.array2QImage(image))
            self.label_2.setPixmap(pixmap)
    
    
    def get_mask_from_image(self, image):
        res = self.Seg.infer(image)
        mask = np.argmax(res, axis=1)
        mask = np.squeeze(mask, 0)
        return mask


    def preprocess(self, image_path=None, ref_path=None):
        # reading
        real_image = cv2.imread(self.image_path)
        if image_path:
            ori_sem = cv2.imread(lib.imgpath_to_labelpath(self.image_path), cv2.IMREAD_GRAYSCALE)
        else:
            input_semantics = self.get_mask_from_image(self.image_path) + 1
        #np.testing.assert_allclose(ori_sem, input_semantics + 1, rtol=1e-03, atol=1e-05)
        reference_image = cv2.imread(self.ref_path)
        if ref_path:
            ori_ref_sem = cv2.imread(lib.imgpath_to_labelpath(self.ref_path), cv2.IMREAD_GRAYSCALE)
        else:
            reference_semantics = self.get_mask_from_image(self.ref_path) + 1
        #np.testing.assert_allclose(ori_ref_sem, reference_semantics + 1, rtol=1e-03, atol=1e-05)
        #produce one-hot labels maps
        #input_semantics = lib.preprocess_with_semantics(input_semantics)
        #reference_semantics = lib.preprocess_with_semantics(reference_semantics)
        reference_image = lib.preprocess_with_images(reference_image)
        input_semantics, reference_semantics = lib.preprocess_input(input_semantics, reference_semantics)
        return input_semantics, reference_image, reference_semantics

   

    def get_result(self, image_path=None, ref_path=None):
        input_semantics, ref_image, ref_semantics = self.preprocess(image_path, ref_path)
        corr_out = self.Corr.infer(input_semantics, ref_image, ref_semantics)
        gen_input = np.concatenate((corr_out[self.Corr.warp_out], input_semantics), axis=1)
        out = self.Gen.infer(gen_input)
        out = lib.postprocess(out)
        return out


    def inference(self):
        # inference for own mask
        result = self.get_result()
        out = cv2.resize(result, dsize=(300, 300), interpolation=Image.BICUBIC)
        pix = QPixmap(self.array2QImage(out))
        self.label_3.setPixmap(pix)
        # inference for ade20k mask
        if os.path.exists(lib.imgpath_to_labelpath(self.ref_path)) and os.path.exists(lib.imgpath_to_labelpath(self.image_path)):
            result = self.get_result(self.image_path, self.ref_path)
            out = cv2.resize(result, dsize=(300, 300), interpolation=Image.BICUBIC)
            pix = QPixmap(self.array2QImage(out))
            self.label_4.setPixmap(pix)
        else:
            self.label_4.setText("There is no corresponding mask (.png)")



def main():
    args = build_argparser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = App(args)
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
