from config.test.config import TestConfig
from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace

class Tests:
    test_configs = []
    """Tests for subtraction from input."""
    def __init__(self) -> None:
        content_base_directory = 'Images/test_images/content_bases'
        style_base_directory = 'Images/test_images/style_bases'
        embedding_directory = 'embeddings'
        self.add_test(TestConfig(
            test_name='axe_1_subtract_from_input_canny_default',
            style_image='axe_1.png',
            content_images=['axe_reference_1.jpg', 'axe_reference_2.jpg', 'axe_reference_3.jpg'],
            embedding='embeddings_axe_1_input_canny_2099.pt',
            feature_detector={'method': canny, 'args': {}},
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
        ))
        self.add_test(TestConfig(
            test_name='axe_1_subtract_from_input_laplace_default',
            style_image='axe_1.png',
            content_images=['axe_reference_1.jpg', 'axe_reference_2.jpg', 'axe_reference_3.jpg'],
            embedding='embeddings_axe_1_input_laplace_2099.pt',
            feature_detector={'method': laplace, 'args': {}},
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
        ))
    
    def add_test(self, test_config: TestConfig):
        self.test_configs.append(test_config)