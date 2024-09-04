from config.test.config import TestConfig
from modules.feature_removal.detectors import canny, difference_of_gaussians, hessian, laplace


class Tests:
    """Collection of TestConfigs that will be sequentially executed by the inference script."""
    test_configs: "list[TestConfig]" = []
    
    def __init__(self) -> None:
        content_base_directory = 'Images/test_images/content_bases'
        style_base_directory = 'Images/test_images/style_bases'
        embedding_directory = 'embeddings/'

        base_test_staff = TestConfig(
            test_name='staff_1_laplace_3099', # override this always
            style_image='staff_1.png',
            content_images=['staff_cc_1.jpg'],
            embedding='embeddings_staff_1_laplace_3099.pt',
            feature_detector={'method': laplace, 'args': {}},
            content_base_directory=content_base_directory,
            style_base_directory=style_base_directory,
            embedding_directory=embedding_directory,
            strengths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            guidance_scales=[3, 6, 9],
            sd_checkpoint='models/sd/control_sd15_canny.pth',
            model_config='configs/stable-diffusion/v1-controlled-inference.yaml',
            controlled=True,
            prompt='*'
        )

        self.add_test(base_test_staff)

        self.add_test(base_test_staff.copy().override({
            'test_name': 'staff_1_canny_3099',
            'embedding': 'embeddings_staff_1_canny_3099.pt',
            'feature_detector': {'method': canny, 'args': {}},
        }))

    
    def add_test(self, test_config: TestConfig):
        self.test_configs.append(test_config)