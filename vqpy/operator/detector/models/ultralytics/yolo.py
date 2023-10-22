from typing import Dict, List
import numpy as np
from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES

from ultralytics import YOLO


# ref: https://github.com/georgia-tech-db/evadb/blob/c637a714c1e68abb5530b20e3ac0d723fe1da3a4/evadb/functions/yolo_object_detector.py
class YoloDetector(DetectorBase):
    cls_names = COCO_CLASSES
    output_fields = ["class_id", "tlbr", "score"]

    def __init__(self, model_path, threshold=0.3, device="gpu"):
        self.model = YOLO(model_path, task="detect")
        self.device = device
        self.threshold = threshold

    def inference(self, img: np.ndarray) -> List[Dict]:
        rets = []
        predictions = self.model.predict(
            img, device=self.device, conf=self.threshold, verbose=False
        )
        for pred_xyxy, pred_conf, pred_cls in zip(
            predictions[0].boxes.xyxy,
            predictions[0].boxes.conf,
            predictions[0].boxes.cls,
        ):  # [0]: only a single frame
            rets.append(
                {
                    "tlbr": np.asarray(pred_xyxy),
                    "score": pred_conf.item(),
                    "class_id": int(pred_cls),
                }
            )
        return rets
