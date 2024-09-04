from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace

class TrainConfig:
    def __init__(self) -> None:
        self.feature_detector = {}
    
    def set_feature_detector_from_string(self, detector: str):
        if detector == 'laplace':
            self.feature_detector = {'method': laplace, 'args': {}}
        elif detector == 'canny':
            self.feature_detector = {'method': canny, 'args': {}}
        elif detector == 'hessian':
            self.feature_detector = {'method': hessian, 'args': {}}
        elif detector == 'dog':
            self.feature_detector = {'method': difference_of_gaussians, 'args': {}}
        else:
            print('No implemented feature detector selected. Training without feature extraction.')