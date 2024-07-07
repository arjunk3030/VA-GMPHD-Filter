import logging
from Constants import DEBUG_MODE, INVALID_OBJECTS
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch


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


def detect_objects(rgbImage, singleView):
    def calculateCls(model, class_idx):
        id_to_index = {
            2: 0,  # Master Chef Can
            3: 1,  # Cracker Box
            4: 2,  # Sugar Box
            5: 3,  # Tomato Soup Can
            6: 4,  # Mustard Bottle
            7: 5,  # Tuna Fish Can
            8: 6,  # Pudding Box
            9: 7,  # Gelatin Box
            10: 8,  # Potted Meat Can
            11: 9,  # Banana
            19: 10,  # Pitcher Base
            21: 11,  # Bleach Cleanser
            24: 12,  # Bowl
            25: 13,  # Mug
            35: 14,  # Power Drill
            36: 15,  # Wood Block
            37: 16,  # Scissors
            40: 17,  # Large Marker
            51: 18,  # Large Clamp
            52: 19,  # Extra Large Clamp
            61: 20,  # Foam Brick
        }
        return id_to_index[int((model.names[int(class_idx)])[:3])]

    model = torch.hub.load("ultralytics/yolov5", "custom", path="yolo-for-ycb/best.pt")

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
            [x, y, z, bb, cls]
            for x, y, z, bb, cls in zip(xs, ys, zs, bbs, clss)
            if cls not in INVALID_OBJECTS
        ]

        if DEBUG_MODE:
            view_image.show()
    return singleView
