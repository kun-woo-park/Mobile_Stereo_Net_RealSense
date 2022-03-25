from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import pyrealsense2 as rs
import torch.backends.cudnn as cudnn
from PIL import Image
from datasets.data_io import get_transform
from models import __models__
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='MSNet2D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--loadckpt', default="./weights/MSNet2D_SF_DS_KITTI2015.ckpt", help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# realsense pipeline
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 800, rs.format.y8, 30)
profile = pipeline.start(config)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def return_processed_img(left_frame, right_frame):
    left_img = Image.fromarray(left_frame).convert('RGB')
    right_img = Image.fromarray(right_frame).convert('RGB')

    # normalize
    processed = get_transform()
    left_img = processed(left_img)
    right_img = processed(right_img)

    return left_img, right_img


def custom_test():
    try:
        model.eval()
        with torch.no_grad():
            while True:
                frames = pipeline.wait_for_frames()
                nir_lf_frame = frames.get_infrared_frame(1)
                nir_rg_frame = frames.get_infrared_frame(2)
                nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
                nir_rg_image = np.asanyarray(nir_rg_frame.get_data())
                nir_lf_image = np.stack([nir_lf_image, nir_lf_image, nir_lf_image], axis=2)
                nir_rg_image = np.stack([nir_rg_image, nir_rg_image, nir_rg_image], axis=2)
                left_img, right_img = return_processed_img(nir_lf_image, nir_rg_image)
                left_img = left_img.reshape(1, left_img.shape[0], left_img.shape[1], left_img.shape[2])
                right_img = right_img.reshape(1, right_img.shape[0], right_img.shape[1], right_img.shape[2])
                disp_ests = model(left_img.cuda(), right_img.cuda())
                disp_est = disp_ests[-1]
                disp_est = tensor2numpy(disp_est)
                disp_est /= 256

                cv2.imshow('Stereo Depth', disp_est.reshape(disp_est.shape[1], disp_est.shape[2]))
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

    finally:
        pipeline.stop()


if __name__ == '__main__':
    # test(args)
    custom_test()
