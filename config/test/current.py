"""
TODO: TESTS FOR EXPERIMENT "SD1.4 boots, ..."
"""

from config.test.config import TestConfig
from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace

from typing import Union

class Tests:
    test_configs: "list[TestConfig]" = []
    """Tests for subtraction from input."""
    def __init__(self) -> None:
        content_base_directory = 'Images/test_images/content_bases'
        style_base_directory = 'Images/test_images/style_bases'
        embedding_directory = 'embeddings/fixed_feature_removal_training/3099'

        ### STAVE TESTS
        base_test_sd15_stave = TestConfig(
            test_name='stave_11_input_laplace_default_sd15_3099_only_control', # override this always
            style_image='stave_11.png',
            content_images=['staff_cc_1.jpg'],
            embedding='embeddings_stave_11_laplace_default_sd15_3099.pt', # override this (always?)
            feature_detector={'method': laplace, 'args': {}},
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
            strengths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            guidance_scales=[3, 6, 9],
            sd_checkpoint='models/sd/control_sd15_canny.pth',
            model_config='configs/stable-diffusion/v1-controlled-inference.yaml',
            controlled=True,
            control_only=False, # Generate image only based on Canny edge map and disregard other content image information.
            prompt='*'
        )

        self.add_test(base_test_sd15_stave)

    
    def add_test(self, test_config: TestConfig):
        self.test_configs.append(test_config)