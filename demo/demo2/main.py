import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QScrollArea, QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt, QPoint
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IECore
sys.path.append("../")
import sample as lib
from queue import Queue


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-s", "--segmentation_network", help="Required. Path to the segmentation network",
    	              default="../../onnx_models/Seg_opset9.onnx", type=str)
    args.add_argument("-c", "--correspondence_network", help="Required. Path to the correspondence_network",
    	              default="../../onnx_models/Corr_opset11.onnx", type=str)
    args.add_argument("-g", "--generate_network", help="Required. Path to the generator",
    	              default="../../onnx_models/Gen_opset11.onnx", type=str)
    return parser


dx = [-1, -1, -1, 0, 0, 1, 1, 1]
dy = [-1, 0, 1, -1, 1, -1, 0, 1]
class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.resize(300, 300)
        pixmap = QtGui.QPixmap(self.size())
        pixmap.fill(Qt.white)
        self.img = pixmap.toImage()
        self.setPixmap(pixmap)
        self.draw = True
        self.was = np.zeros((300, 300))
        print(self.pixmap())
        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')   
        self.pen_width = 3     

    def clean(self):
        self.pixmap().fill(Qt.white)
        self.update()
    
    def fill(self):
        self.draw = False

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)
    

    def set_pen_width(self, width):
        self.pen_width = width
    
    def check(self, x, y):
        if (x < 0) or (y < 0) or x >= self.width() or y >= self.height():
            return False
        col = QColor(self.pixmap().toImage().pixel(x,y)).getRgbF()
        if col == (1.0, 1.0, 1.0, 1.0):
            return True
        return False
    
    def color_bfs(self, x, y, painter):
        q = Queue()
        q.put((x, y))
        self.was[x][y] = 1
        print("begin")
        while not q.empty():
            x, y = q.get()
            #self.img.setPixelColor(QPoint(x, y), self.pen_color)
            painter.drawPoint(QPoint(x, y))
            self.update()
            for i in range(8):
                _x = x + dx[i]
                _y = y + dy[i]
                if self.check(_x, _y) and self.was[_x][_y] == 0:
                    q.put((_x, _y))
                    self.was[_x][_y] = 1


    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(self.pen_width)
        p.setColor(self.pen_color)
        painter.setPen(p)
        if self.draw:
            painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
            painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mousePressEvent(self, e):
        if not self.draw:
            painter = QtGui.QPainter(self.pixmap())
            p = painter.pen()
            p.setWidth(1)
            p.setColor(self.pen_color)
            painter.setPen(p)
            self.was[:] = 0
            self.draw = True
            self.color_bfs(e.x(), e.y(), painter)
            painter.end()
            self.update()
            print("end")
        

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

# map for 15 items and rubber
COLORS = {
    'wall': '#787878', 'sky': '#06E6E6', 'tree': '#04C803', 'road': '#8C8C8C', 'door': '#08FF33',
    'person': '#96053D', 'ground': '#787846', 'water': '#3DE6FA', 'sea': '#0907E6', 'building': '#B47878',
    'grass': '#04FA07', 'plants': '#CCFF04', 'car': '#0066C8', 'house': '#FF09E0', 'waterfall': '#00E0FF',
    'rubber': '#ffffff'
}

class QPaletteButton(QtWidgets.QPushButton):

    def __init__(self, color):
        super().__init__()
        self.setFixedSize(QtCore.QSize(55,35))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, args):
        super().__init__()
        '''
        self.core = IECore()
        self.Corr = lib.correspondence_model(args.correspondence_network, self.core)
        self.Gen = lib.generate_model(args.generate_network, self.core)
        self.Seg = lib.segmentation_model(args.segmentation_network, self.core)
        '''
        self.resize(922, 696)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        # grid 
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        # canvas on grid
        self.canvas = Canvas()
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.canvas)
        scrollarea = QScrollArea()
        palette = QtWidgets.QHBoxLayout()
        
        lst = QtWidgets.QWidget()
        lst.adjustSize()
        lst.setBackgroundRole(QPalette.Dark)
        lst.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        lst.setLayout(palette)
        #scroll.setWidget(palette)
        self.add_palette_buttons(palette)
        scroll = QScrollArea()
        scroll.setWidget(lst)
        scroll.setMaximumWidth(self.canvas.width())
        scroll.setMaximumHeight(65)
        l.addLayout(palette)
        self.gridLayout.addWidget(self.canvas, 0, 0, 1, 1)
        self.gridLayout.addWidget(scroll, 1, 0, 1, 1)
        #clear button
        self.clearButton = QtWidgets.QPushButton(self.centralwidget)
        self.clearButton.setObjectName("clearButton")
        self.clearButton.setText("Clear")
        self.clearButton.clicked.connect(self.canvas.clean)
        self.gridLayout.addWidget(self.clearButton, 2, 0, 1, 1)
        #fill button
        self.fillButton = QtWidgets.QPushButton(self.centralwidget)
        self.fillButton.setObjectName("fillButton")
        self.fillButton.setText("Fill")
        self.fillButton.clicked.connect(self.canvas.fill)
        self.gridLayout.addWidget(self.fillButton, 3, 0, 1, 1)
        #label reference and button
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setText("Reference image")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Open reference")
        self.pushButton.clicked.connect(self.get_reference)
        self.gridLayout.addWidget(self.pushButton, 1, 1, 1, 1)
        #label result and button
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setAlignment(QtCore.Qt.AlignCenter)
        self.result.setText("Result")
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setObjectName("pushButton2")
        self.pushButton2.setText("Get result")
        self.pushButton2.clicked.connect(self.inference)
        self.gridLayout.addWidget(self.result, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.pushButton2, 1, 2, 1, 1)
        self.setCentralWidget(self.centralwidget)


        #self.retranslateUi()
    def array2QImage(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        image = np.require(image, np.uint8, 'C')
        qImg = QImage(image, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg
    
    def QPixmapToCvMat(self, pixmap):
        ## Get the size of the current pixmap
        size = pixmap.size()
        h = size.width()
        w = size.height()
    
        ## Get the QImage Item and convert it to a byte string
        qimg = pixmap.toImage()
        b = qimg.bits()
        b.setsize(h * w * 4)
        ## Using the np.frombuffer function to convert the byte string into an np array
        img = np.frombuffer(b, dtype=np.uint8).reshape((w,h,4))
        return img

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "CoCosNet Demo"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p>Reference image</p></body></html>"))
        
    

    def draw(self, color, width):
        self.canvas.draw = True
        self.canvas.set_pen_color(color)
        self.canvas.set_pen_width(width)


    def add_palette_buttons(self, layout):
        for key, c in COLORS.items():
            b = QPaletteButton(c)
            b.setText(key)
            if c == '#ffffff':
                b.pressed.connect(lambda c=c, w=15: self.draw(c, w))
            else:
                b.pressed.connect(lambda c=c, w=3: self.draw(c,w))
            layout.addWidget(b)

    def get_reference(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image File")

        if file_name:
            self.ref_path = file_name[0]
            image = cv2.imread(file_name[0])
            image = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(self.array2QImage(image))
            self.label.setPixmap(pixmap)
    
    def get_mask_from_image(self, image):
        res = self.Seg.infer(image)
        mask = np.argmax(res, axis=1)
        mask = np.squeeze(mask, 0)
        return mask


    def preprocess(self):
        # reading
        real_image = self.QPixmapToCvMat(self.canvas.pixmap())
        input_semantics = self.get_mask_from_image(real_image) + 1
        reference_image = cv2.imread(self.ref_path)
        reference_semantics = self.get_mask_from_image(self.ref_path) + 1
        #np.testing.assert_allclose(ori_ref_sem, reference_semantics + 1, rtol=1e-03, atol=1e-05)
        #produce one-hot labels maps
        input_semantics = lib.preprocess_with_semantics(input_semantics)
        reference_semantics = lib.preprocess_with_semantics(reference_semantics)
        reference_image = lib.preprocess_with_images(reference_image)
        input_semantics, reference_semantics = lib.preprocess_input(input_semantics, reference_semantics)
        return input_semantics, reference_image, reference_semantics

   

    def get_result(self):
        input_semantics, ref_image, ref_semantics = self.preprocess()
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
        self.result.setPixmap(pix)



def main():
    args = build_argparser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()