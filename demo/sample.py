import numpy as np
import sys
import cv2
from PIL import Image
import torch
from torchvision import transforms
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-ir", "--input_real", help="Required. Path to real image and its semantics", required=True,
                      type=str)
    args.add_argument("-is", "--input_reference", help="Required. Path to the reference image and its semantics",
                      required=True, type=str)
    args.add_argument("-c", "--correspondence_network", help="Required. Path to the correspondence_network",
    	              default="onnx_models/Corr_opset11.onnx", type=str)
    args.add_argument("-g", "--generate_network", help="Required. Path to the generator",
    	              default="onnx_models/Gen_opset11.onnx", type=str)
    args.add_argument("-o", "--output_path", help="Required. Path to the output folder",
    	              default="output/demo_results/res.jpg", type=str)
    return parser


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class correspondence_model:
    def __init__(self, path, core):
        self.network = core.read_network(path)
        self.input_keys = list(self.network.input_info.keys())
        self.input_semantics = self.input_keys[0]
        self.reference_image = self.input_keys[1]
        self.reference_semantics = self.input_keys[2]
        self.output_keys = list(self.network.outputs.keys())
        self.warp_out = self.output_keys[0]
        self.warp_mask = self.output_keys[1]
        self.exec_net = core.load_network(network=self.network, device_name="CPU")
        print("Correspondence model create!")


    def infer(self, input_sem, ref_image, ref_semantic):
        input = {}
        input[self.input_semantics] = input_sem
        input[self.reference_image] = ref_image
        input[self.reference_semantics] = ref_semantic
        out = self.exec_net.infer(inputs=input)
        return out


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
    img = img.numpy()
    return np.array([img])


def preprocess_seg(image_path):
	img = Image.open(image_path).convert('RGB')
	w, h = img.size
	img = img.resize(size=(320, 320), resample=Image.BILINEAR)
	img = img_transform(img)
	return img


class segmentation_model:
    def __init__(self, path, core):
        self.network = core.read_network(path)
        self.input_blob = next(iter(self.network.input_info))
        self.output_blob = next(iter(self.network.outputs))
        self.exec_net = core.load_network(network=self.network, device_name="CPU")
        print("Segmentation model create!")

    def infer(self, image_path):
        image = preprocess_seg(image_path)
        out = self.exec_net.infer(inputs={self.input_blob: image})
        return out[self.output_blob]


class generate_model:
    def __init__(self, path, core):
        self.network = core.read_network(path)
        self.input_keys = list(self.network.input_info.keys())
        self.input_blob = self.input_keys[0]
        self.output_keys = list(self.network.outputs.keys())
        self.output_blob = self.output_keys[0]
        self.exec_net = core.load_network(network=self.network, device_name="CPU")
        print("Generative model create!")

    def infer(self, input):
        inp = {}
        inp[self.input_blob] = input
        out = self.exec_net.infer(inputs=inp)
        return out[self.output_blob]


def imgpath_to_labelpath(path):
    return path.replace(".jpg", ".png")


def preprocess_with_semantics(sem):
    print(sem.shape)
    sem = cv2.resize(sem, dsize=(256, 256), interpolation=Image.NEAREST)
    sem[sem == 255] = 150
    return sem


def preprocess_with_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(256, 256), interpolation=Image.BICUBIC)
    image = np.transpose(image, (2, 0, 1))
    image = np.divide(image, 255)
    image = np.subtract(image, 0.5)
    image = np.divide(image, 0.5)
    return image


def scatter(source):
	out = np.zeros((1, 151, 256, 256))
	h, w = source.shape
	for i in range(h):
		for j in range(w):
			out[0][source[i][j]][i][j] = 1
	return out


def preprocess_input(label, label_ref):
# create one-hot label map
    input_semantics = scatter(label)
    ref_semantics = scatter(label_ref)

    return input_semantics, ref_semantics


def inference(Corr, Gen, input_semantics, ref_image, ref_semantics):
    corr_out = Corr.infer(input_semantics, ref_image, ref_semantics)
    gen_input = np.concatenate((corr_out[Corr.warp_out], input_semantics), axis=1)
    out = Gen.infer(gen_input)
    return out

def postprocess(result):
    result = np.squeeze(result)
    result = np.add(result, 1)
    result = np.divide(result, 2)
    result = np.multiply(result, 255)
    result = np.transpose(result, (1, 2, 0))
    return result

def save_results(result, path):
    try:
        print(result.shape)
        result = np.squeeze(result)
        result = np.add(result, 1)
        result = np.divide(result, 2)
        result = np.multiply(result, 255)
        result = np.transpose(result, (1, 2, 0))
        result= cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print(result.shape)
        cv2.imwrite(path, result)
    except OSError as err:
        print(err)


def main():
	#initialization
    args = build_argparser().parse_args()
    real_image = cv2.imread(args.input_real)
    input_semantics = cv2.imread(imgpath_to_labelpath(args.input_real), cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(args.input_reference)
    reference_semantics = cv2.imread(imgpath_to_labelpath(args.input_reference), cv2.IMREAD_GRAYSCALE)
    # get input
    input_semantics = preprocess_with_semantics(input_semantics)
    reference_semantics = preprocess_with_semantics(reference_semantics)
    reference_image = preprocess_with_images(reference_image)
    # one-hot label maps
    input_semantics, reference_semantics = preprocess_input(input_semantics, reference_semantics)
    #inference
    core = IECore()
    Corr = correspondence_model(args.correspondence_network, core)
    Gen = generate_model(args.generate_network, core)
    result = inference(Corr, Gen, input_semantics, reference_image, reference_semantics)
    print(result)
    #save results
    save_results(result, args.output_path)
    


if __name__ == '__main__':
    sys.exit(main() or 0)