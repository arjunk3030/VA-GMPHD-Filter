import logging
from PIL import Image, ImageDraw
import numpy as np
from util_files.TrajectorySettings import DetectedObject, View
from util_files.config_params import DEBUG_MODE
from util_files.object_parameters import ID_TO_INDEX
    
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

def detect_objects(view: View, model):
    """
    Runs object detection on a View and fills its objects list.
    """
    def calculate_cls(model, class_idx):
        return ID_TO_INDEX[int((model.names[int(class_idx)])[:3])]

    rgb_image = transform_background_color(view.rgb)
    result = model(rgb_image)
    view_image = Image.fromarray(result.render()[0])

    detected_objects = []

    for det in result.xywh:
        for *box, class_idx in det:
            cls_id = calculate_cls(model, class_idx)
            x_center, y_center, width, height, conf = [b.item() for b in box]
            z = view.depth[round(y_center), round(x_center)]
            bbox = [x_center, y_center, width, height]

            # Log detection
            logging.info(
                "A %s detected at location (%s, %s) on step %d",
                model.names[int(class_idx)],
                x_center,
                y_center,
                view.step,
            )

            # Draw for visualization
            draw = ImageDraw.Draw(view_image)
            draw.ellipse(
                [(x_center - 2, y_center - 2), (x_center + 2, y_center + 2)],
                fill=(255, 0, 0),
            )

            detected_objects.append(DetectedObject(x=x_center, y=y_center, z=z, bbox=bbox, cls=cls_id))

    return detected_objects