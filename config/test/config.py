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