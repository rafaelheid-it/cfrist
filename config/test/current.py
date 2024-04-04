from config.test.config import TestConfig
from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace

class Tests:
    test_configs = []
    """Tests for subtraction from input."""
    def __init__(self) -> None:
        content_base_directory = 'Images/test_images/content_bases'
        style_base_directory = 'Images/test_images/style_bases'
        embedding_directory = 'embeddings'
        base_test = TestConfig(
            test_name='axe_2_subtract_from_input_canny_default', # override this always
            style_image='axe_2.png',
            content_images=['axe_reference_1.jpg', 'axe_reference_3.jpg', 'axe_reference_5.jpg', 'axe_reference_7.jpg'],
            embedding='embeddings_axe_2_canny_default_2099.pt', # override this (always?)
            feature_detector={'method': canny, 'args': {}},
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
            strengths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            guidance_scales=[2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        self.add_test(base_test.copy())
        self.add_test(base_test.copy().override({
            'test_name': 'axe_2_subtract_from_input_dog_default',
            'embedding': 'embeddings_axe_2_dog_default_2099.pt',
            'feature_detector': {'method': difference_of_gaussians, 'args': {}}
        }))
        self.add_test(base_test.copy().override({
            'test_name': 'axe_2_subtract_from_input_hessian_default',
            'embedding': 'embeddings_axe_2_hessian_default_2099.pt',
            'feature_detector': {'method': hessian, 'args': {}}
        }))
        self.add_test(base_test.copy().override({
            'test_name': 'axe_2_subtract_from_input_laplace_default',
            'embedding': 'embeddings_axe_2_laplace_default_2099.pt',
            'feature_detector': {'method': laplace, 'args': {}}
        }))
        
        # base_test = TestConfig(
        #     test_name='stave_7_subtract_from_input_canny_default', # override this always
        #     style_image='stave_07.png',
        #     content_images=[], # TODO: Add stave content base images.
        #     embedding='embeddings_stave_7_canny_default_2099.pt', # override this (always?)
        #     feature_detector={'method': canny, 'args': {}},
        #     content_base_directory=content_base_directory,
        #     style_base_directory=style_base_directory,
        #     embedding_directory=embedding_directory,
        #     strengths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     guidance_scales=[2, 3, 4, 5, 6, 7, 8, 9, 10]
        # )
        # self.add_test(base_test.copy())
    
    def add_test(self, test_config: TestConfig):
        self.test_configs.append(test_config)