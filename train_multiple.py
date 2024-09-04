import subprocess

feature_extractors = ['laplace']
style_images = ['axe_1', 'armor_1', 'staff_1']
sd_config = 'configs/stable-diffusion/v1-finetune.yaml'
sd_checkpoint = './models/sd/v1-5-pruned-emaonly.ckpt'

for feature_extractor in feature_extractors:
    for style_image in style_images:
        training_name = style_image + '_' + feature_extractor + '_' + 'training_run'
        data_root = 'Images/train_images/' + style_image
        process = subprocess.run([
            'python', 'main.py', 
            '--base', sd_config,
            '-t',
            '--actual_resume', sd_checkpoint,
            '-n', training_name,
            '--feature_extractor', feature_extractor,
            '--gpus', '0,',
            '--data_root', data_root
        ])