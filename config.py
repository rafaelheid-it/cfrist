from dataclasses import dataclass
from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace
from detectors import canny, difference_of_gaussians, hessian, laplace

@dataclass
class TestConfig:
    """TestConfig for original model."""
    content_base_directory = 'Images/test_images/content_bases'
    style_base_directory = 'Images/test_images/style_bases'
    embedding_directory = 'embeddings'
    style_base_embedding_map = {
        'axe_1.png': 'embeddings_axe_1_original_2099.pt'
    }
    detector = {
        "method": canny,
        "kwargs": {}
    }