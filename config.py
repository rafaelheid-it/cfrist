from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace
from pathlib import Path

class TestConfig:
    """TestConfig for original model."""
    def __init__(
            self,
            test_name: str,
            style_image: str,
            content_images: list,
            embedding: str,
            content_base_directory: str = '',
            style_base_directory: str = '',
            embedding_directory: str = '',
            feature_detector: dict = None,
            strengths: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            guidance_scales: list = list(range(2, 11))
        ) -> None:
        self.test_name = test_name
        self.style_image = style_image
        self.content_images = content_images
        self.embedding = embedding
        
        self.content_base_directory = content_base_directory
        self.style_base_directory = style_base_directory
        self.embedding_directory = embedding_directory
        self.feature_detector = feature_detector

        self.strengths = strengths
        self.guidance_scales = guidance_scales

    @property
    def embedding_path(self):
        return str(Path(self.embedding_directory, self.embedding))
    
    @property
    def style_image_path(self):
        return str(Path(self.style_base_directory, self.style_image))
    
    @property
    def content_image_paths(self):
        return [str(Path(self.content_base_directory, content_image)) for content_image in self.content_images]
    

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
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
            feature_detector={'method': canny, 'args': {}}
        ))
        self.add_test(TestConfig(
            test_name='axe_1_subtract_from_input_laplace_default',
            style_image='axe_1.png',
            content_images=['axe_reference_1.jpg', 'axe_reference_2.jpg', 'axe_reference_3.jpg'],
            embedding='embeddings_axe_1_input_laplace_2099.pt',
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
            feature_detector={'method': laplace, 'args': {}}
        ))
    
    def add_test(self, test_config: TestConfig):
        self.test_configs.append(test_config)

class TrainConfig:
    feature_detector = {'method': canny, 'args': {}}

class GlobalConfig:
    config = TrainConfig()

    @staticmethod
    def set(config):
        GlobalConfig.config = config