import argparse
import logging
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
from Constants import DEBUG_MODE
import torch
import torch.nn as nn
import ssl
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from DFP.lib.network import PoseNet, PoseRefineNet
from DFP.lib.transformations import (
    euler_matrix,
    quaternion_matrix,
    quaternion_from_matrix,
)
from torchvision import models, transforms

# Our file


class DenseProcessor:
    def __init__(self, cam_intrinsic, model_config, rgb, depth):
        (self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy) = cam_intrinsic
        self.cam_scale = 1
        # np_image = np.array(rgb)
        # float_image = np_image.astype(np.float32)
        self.img = rgb
        self.depth = depth
        (
            self.num_points,
            self.num_obj,
            self.img_length,  # renderer height
            self.img_width,  # renderer width
        ) = model_config
        self.bs = 1
        self.iteration = 2
        self.border_list = [
            -1,
            40,
            80,
            120,
            160,
            200,
            240,
            280,
            320,
            360,
            400,
            440,
            480,
            520,
            560,
            600,
            640,
            680,
        ]
        self.initialize_model()

    def initialize_model(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model",
            type=str,
            default="DFP/trained_models/ycb/pose_model_26_0.012863246640872631.pth",
            help="model",
        )
        parser.add_argument(
            "--refine-model",
            type=str,
            default="DFP/trained_models/ycb/pose_refine_model_69_0.009449292959118935.pth",
            help="refine model",
        )
        opt = parser.parse_args()

        # Instantiate PoseNet and PoseRefineNet models on CPU
        estimator = PoseNet(num_points=self.num_points, num_obj=self.num_obj)
        estimator.load_state_dict(
            torch.load(opt.model, map_location=torch.device("cpu"))
        )
        estimator.eval()

        refiner = PoseRefineNet(num_points=self.num_points, num_obj=self.num_obj)
        refiner.load_state_dict(
            torch.load(opt.refine_model, map_location=torch.device("cpu"))
        )
        refiner.eval()

        self.estimator = estimator
        self.refiner = refiner

    # def get_bbox(self, bounded_box):
    #     x_center, y_center, width, height = bounded_box

    #     rmin = int(y_center - height / 2) + 1
    #     rmax = int(y_center + height / 2) - 1
    #     cmin = int(x_center - width / 2) + 1
    #     cmax = int(x_center + width / 2) - 1

    #     center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

    #     r_b = rmax - rmin
    #     c_b = cmax - cmin

    #     rmin = max(center[0] - int(r_b / 2), 0)
    #     rmax = min(center[0] + int(r_b / 2), self.img_width)
    #     cmin = max(center[1] - int(c_b / 2), 0)
    #     cmax = min(center[1] + int(c_b / 2), self.img_length)

    #     return rmin, rmax, cmin, cmax

    def get_bbox(self, bounded_box):
        x_center, y_center, width, height = bounded_box
        rmin = int(y_center - height / 2) + 1
        rmax = int(y_center + height / 2) - 1
        cmin = int(x_center - width / 2) + 1
        cmax = int(x_center + width / 2) - 1

        r_b = rmax - rmin
        c_b = cmax - cmin

        for tt in range(len(self.border_list) - 1):
            if r_b > self.border_list[tt] and r_b < self.border_list[tt + 1]:
                r_b = self.border_list[tt + 1]
                break

        for tt in range(len(self.border_list) - 1):
            if c_b > self.border_list[tt] and c_b < self.border_list[tt + 1]:
                c_b = self.border_list[tt + 1]
                break

        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

        rmin = max(center[0] - int(r_b / 2), 0)
        rmax = min(center[0] + int(r_b / 2), self.img_width)
        cmin = max(center[1] - int(c_b / 2), 0)
        cmax = min(center[1] + int(c_b / 2), self.img_length)

        return rmin, rmax, cmin, cmax

    def process_data(self, bounded_box, id):
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        try:
            rmin, rmax, cmin, cmax = self.get_bbox(bounded_box)
            img_masked = np.array(self.img)[:, :, :3]
            img_masked = img_masked[rmin:rmax, cmin:cmax, :]

            img_pil = Image.fromarray(np.uint8(img_masked))

            if DEBUG_MODE:
                # Save or display the image
                img_pil.show()

            mask = ma.getmaskarray(ma.masked_not_equal(self.depth, 0))
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) > self.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[: self.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_points - len(choose)), "wrap")

            xmap = np.array(
                [[j for i in range(self.img_width)] for j in range(self.img_length)]
            )
            ymap = np.array(
                [[i for i in range(self.img_width)] for j in range(self.img_length)]
            )

            depth_masked = (
                self.depth[rmin:rmax, cmin:cmax]
                .flatten()[choose][:, np.newaxis]
                .astype(np.float32)
            )
            xmap_masked = (
                xmap[rmin:rmax, cmin:cmax]
                .flatten()[choose][:, np.newaxis]
                .astype(np.float32)
            )
            ymap_masked = (
                ymap[rmin:rmax, cmin:cmax]
                .flatten()[choose][:, np.newaxis]
                .astype(np.float32)
            )
            choose = np.array([choose])

            pt2 = depth_masked / self.cam_scale
            pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
            pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(self.img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([id])

            cloud = Variable(cloud)
            choose = Variable(choose)
            img_masked = Variable(img_masked)
            index = Variable(index)

            cloud = cloud.view(1, self.num_points, 3)
            img_masked = img_masked.view(
                1, 3, img_masked.size()[1], img_masked.size()[2]
            )

            pred_r, pred_t, pred_c, emb = self.estimator(
                img_masked, cloud, choose, index
            )
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

            pred_c = pred_c.view(self.bs, self.num_points)
            how_max, which_max = torch.max(pred_c, 1)
            logging.info("Confidence is 6d pose estimation: %f", how_max)
            pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
            points = cloud.view(self.bs * self.num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).data.numpy()
            my_pred = np.append(my_r, my_t)

            for ite in range(0, self.iteration):
                T = (
                    Variable(torch.from_numpy(my_t.astype(np.float32)))
                    .view(1, 3)
                    .repeat(self.num_points, 1)
                    .contiguous()
                    .view(1, self.num_points, 3)
                )
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).view(
                    1, 3, 3
                )
                my_mat[0:3, 3] = my_t

                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = self.refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).data.numpy()
                my_t_2 = pred_t.view(-1).data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array(
                    [my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]]
                )

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final
        except ZeroDivisionError:
            logging.log("error")
            my_pred.append([-1.0 for i in range(7)])
        return my_pred