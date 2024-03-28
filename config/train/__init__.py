from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace

class TrainConfig:
    feature_detector = {'method': canny, 'args': {}}