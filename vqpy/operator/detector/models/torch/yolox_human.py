"""
Based on demo implementation in Megvii YOLOX repo
The YOLOX detector for object detection
"""

from typing import Dict, List

import numpy as np
import torch
from loguru import logger
import cv2
from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES

from yolox.exp import Exp as MyExp
from yolox.utils import postprocess, fuse_model
from yolox.utils.model_utils import get_model_info


# Exp from https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/exps/example/mot/yolox_s_mix_det.py   # noqa: E501
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_s_mix_det"
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.random_size = (12, 26)
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0


# preproc from older versions of YOLOX
def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class YOLOXHumanDetector(DetectorBase):
    """The YOLOX detector for object detection"""

    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]

    def __init__(self, model_path, device="gpu", fp16=True):
        # TODO: start a new process handling this
        exp = Exp()

        model = exp.get_model().to(
            torch.device("cuda" if device == "gpu" else "cpu")
        )
        model_info = get_model_info(model, exp.test_size)
        logger.info(f"Model Summary: {model_info}")
        if device == "gpu":
            model.cuda()
        model.eval()

        logger.info("loading checkpoint")
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        model = fuse_model(model)
        if fp16:
            model.half()

        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        # self.preproc = ValTransform(legacy=False)
        self.preproc = preproc
        self.postproc = postprocess
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img) -> List[Dict]:
        ratio = min(
            self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1]
        )

        img, _ = self.preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = self.postproc(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )

        outputs = outputs[0]
        if outputs is None:
            return []
        bboxes = (outputs[:, 0:4] / ratio).cpu()
        scores = (outputs[:, 4:5] * outputs[:, 5:6]).cpu()
        cls = outputs[:, 6:7].cpu()

        rets = []
        for (tlbr, score, class_id) in zip(bboxes, scores, cls):
            rets.append(
                {
                    "tlbr": np.asarray(tlbr),
                    "score": score.item(),
                    "class_id": int(class_id.item()),
                }
            )
        return rets
