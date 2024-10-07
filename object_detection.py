import logging
from queue import Queue
from Constants import DEBUG_MODE
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import Constants


def transform_background_color(image_np, color=(255, 255, 255, 0)):
    image_pil = Image.fromarray(image_np)
    image_pil = image_pil.convert("RGBA")

    pixel_data = image_pil.getdata()

    new_image_data = []
    for item in pixel_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_image_data.append(color)
        else:
            new_image_data.append(item)

    image_pil.putdata(new_image_data)
    image_pil = image_pil.convert("RGB")

    return image_pil


def detect_objects(rgbImage, singleView, model):
    def calculateCls(model, class_idx):
        return Constants.ID_TO_INDEX[int((model.names[int(class_idx)])[:3])]

    rgb_image = transform_background_color(rgbImage)

    result = model(rgb_image)

    xs, ys, zs, bbs, clss = [], [], [], [], []

    view_image = Image.fromarray((result.render()[0]))
    for j, det in enumerate(result.xywh):
        for *box, class_idx in det:
            clss.append(calculateCls(model, class_idx))

            x_center, y_center, width, height, conf = box
            x_center = x_center.item()
            y_center = y_center.item()
            width = width.item()
            height = height.item()
            bbs.append([x_center, y_center, width, height])

            logging.info(
                "A %s detected at location (%s, %s) on step %d",
                model.names[int(class_idx)],
                x_center,
                y_center,
                singleView["step"],
            )

            draw = ImageDraw.Draw(view_image)
            draw.ellipse(
                [(x_center - 2, y_center - 2), (x_center + 2, y_center + 2)],
                fill=(255, 0, 0),
            )

            xs.append(x_center)
            ys.append(y_center)
            zs.append(singleView["depth_img"][round(y_center), round(x_center)])

        singleView["rgb_img"] = rgb_image
        singleView["objects"] = [
            [x, y, z, bb, cls] for x, y, z, bb, cls in zip(xs, ys, zs, bbs, clss)
        ]

        if DEBUG_MODE:
            view_image.show()  # TODO:L ADD BACK
    return singleView
