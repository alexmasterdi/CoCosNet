import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
import design
import numpy as np
from PIL import Image
import sample as lib
import cv2
from openvino.inference_engine import IECore

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.get_image)
        self.pushButton_2.clicked.connect(self.get_reference)
        self.pushButton_3.clicked.connect(self.inference)
        self.core = IECore()
        self.Corr = lib.correspondence_model("../onnx_models/Corr_opset11.onnx", self.core)
        self.Gen = lib.generate_model("../onnx_models/Gen_opset11.onnx", self.core)
    

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
            image = cv2.resize(image, dsize=(512, 512), interpolation=Image.BICUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(self.array2QImage(image))
            self.label.setPixmap(pixmap)


    def get_reference(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open Reference File")

        if file_name: 
            self.ref_path = file_name[0]
            image = cv2.imread(file_name[0])
            image = cv2.resize(image, dsize=(512, 512), interpolation=Image.BICUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(self.array2QImage(image))
            self.label_2.setPixmap(pixmap)
    

    def preprocess(self):
        real_image = cv2.imread(self.image_path)
        input_semantics = cv2.imread(lib.imgpath_to_labelpath(self.image_path), cv2.IMREAD_GRAYSCALE)
        reference_image = cv2.imread(self.ref_path)
        reference_semantics = cv2.imread(lib.imgpath_to_labelpath(self.ref_path), cv2.IMREAD_GRAYSCALE)
        input_semantics = lib.preprocess_with_semantics(input_semantics)
        reference_semantics = lib.preprocess_with_semantics(reference_semantics)
        reference_image = lib.preprocess_with_images(reference_image)
        input_semantics, reference_semantics = lib.preprocess_input(input_semantics, reference_semantics)
        return input_semantics, reference_image, reference_semantics

   
    def inference(self):
        input_semantics, ref_image, ref_semantics = self.preprocess()
        corr_out = self.Corr.infer(input_semantics, ref_image, ref_semantics)
        gen_input = np.concatenate((corr_out[self.Corr.warp_out], input_semantics), axis=1)
        out = self.Gen.infer(gen_input)
        out = lib.postprocess(out)
        print(out.shape)
        print(type(out))
        out = cv2.resize(out, dsize=(512, 512), interpolation=Image.BICUBIC)
        pix = QPixmap(self.array2QImage(out))
        self.label_3.setPixmap(pix)



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()