import logging
import numpy as np

from cvlib.object_detection import populate_class_labels, draw_bbox, detect_common_objects

from traffic_monitor.services.detectors.detector_abstract import DetectorAbstract

logger = logging.getLogger('detector')


class DetectorCVlib(DetectorAbstract):
    """
    Implementation of DetectorAbstract.  This implementation is from the OpenCV
    implementation of object instance detection.

    https://github.com/arunponnusamy/cvlib

    Yolov4 cfg and weights are available at: https://github.com/AlexeyAB/darknet

    Supports models:
        yolov3-tiny
        yolov3

    Requires that .cfg file and .weights files are in ~/.cvlib/object_detection/yolo/yolov3
    """

    def __init__(self, monitor_config: dict):
        DetectorAbstract.__init__(self, monitor_config)
        self.detector_name: str = monitor_config.get('detector_name')
        self.detector_model: str = monitor_config.get('detector_model')
        self.detector_confidence: float = monitor_config.get('detector_confidence')
        # note that colors in cvlib uses BGR not RGB colors
        self.bgr_colors = np.float64([monitor_config.get('class_colors').get(o)[::-1] for o in populate_class_labels()])

    def set_detector_value(self, kwargs_list: list):
        """ Only allow changes to confidence or the model """
        try:
            for kwargs in kwargs_list:
                field = kwargs.get('field')
                value = kwargs.get('value')
                if field in ['detector_confidence', 'detector_model']:
                    logger.info(f"{self.detector_name}: setting value: {field}: {value}")
                    self.monitor_config[field] = value
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Error setting value: {e}")

    def detect(self, frame: np.array) -> (np.array, list):
        # colors is a list of BGR values in a list ([[#b,#g,#r],[#b,#g,#r], ... ])
        try:
            bbox, labels, conf = detect_common_objects(frame, confidence=self.detector_confidence, model=self.detector_model)
            frame = draw_bbox(img=frame, bbox=bbox, labels=labels, confidence=conf, write_conf=False, colors=self.bgr_colors)
            return frame, labels
        except Exception as e:
            logger.error(f"{self.__class__.__name__} Exception: {e}")

    @classmethod
    def get_trained_objects(cls) -> list:
        return populate_class_labels()
