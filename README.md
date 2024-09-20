<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<!-- <div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div> -->



<!-- TABLE OF CONTENTS -->
<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">CAST</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->

This code depends heavily on [InST](https://github.com/zyxElsa/InST) and [ControlNet](https://github.com/lllyasviel/ControlNet) and their dependencies. Many thanks for openly providing implementations!

<!-- ABOUT THE PROJECT -->
## CFRIST (Controlled Feature-Removed Inversion-based Style Transfer)
Thesis title: _Reducing object distortion in inversion-based image style transfer_

<!-- ![teaser](./Images/teaser.png) -->
<!--![teaser](./Images/teaser.png)-->

The video game industry has been the biggest market in the entertainment sector for the past decades. As an important factor in their success, art as a medium used in a game can make the player feel a multitude of emotions and transport them into a different reality. The generation of such images can be both time and cost consuming, especially for independent developers without the backing of some organization. Image style transfer can be a great way to generate images for a game in different contexts but similar feel, by applying the style of a specific reference image onto an image with the desired content structure. One such method is inversion-based style transfer, that can learn the style in the reference image in about 20 minutes, by encoding the style information as the underlying diffusion model's text conditioning. Due to the method tending to deform objects with longer stylization rounds towards objects from the style reference, we propose Controlled Feature-Removed Inversion-based Style Transfer (CFRIST), a two-way scheme that enhances generated content structures in the stylized image. We firstly enrich the textual inversion process by removing object shape information from the style image while learning its feature vector. We secondly enforce the content image's object structure by controlling the generation process via a pre-trained ControlNet. The resulting method has the same training time as the original inversion-based style transfer, but synthesizes images where the content image's object structure is still intact even after long rounds of stylization.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ### Built With -->
<!-- 
This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [Next.js](https://nextjs.org/)
* [React.js](https://reactjs.org/)
* [Vue.js](https://vuejs.org/)
* [Angular](https://angular.io/)
* [Svelte](https://svelte.dev/)
* [Laravel](https://laravel.com)
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)

<p align="right">(<a href="#top">back to top</a>)</p>
 -->


<!-- GETTING STARTED -->
## Getting Started

### Clone repository

   Clone the repository with its ControlNet dependency as submodule.
   ```sh
   git clone --recurse-submodules https://github.com/rafaelheid-it/cfrist.git
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Create Python environment
Please be aware, that you need to have some version [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) installed for this to work.

For packages, see environment.yaml.

  ```sh
  conda env create -f environment.yaml
  conda activate cfrist
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Train
  Download the pretrained [Stable Diffusion Model](https://huggingface.co/benjamin-paine/stable-diffusion-v1-5) and save it at `./models/sd/v1-5-pruned-emaonly.ckpt`.

   Train CFRIST:
   ```sh
   python main.py \
   --base configs/stable-diffusion/v1-finetune.yaml \
    -t \
    --actual_resume ./models/sd/v1-5-pruned-emaonly.ckpt \
    -n <run_name> \
    --gpus 0, \
    --data_root /path/to/directory/with/images \
    --feature_extractor laplace \
   ```
   
   See `configs/stable-diffusion/v1-finetune.yaml` for more options.

   The command is also available in `train_command.sh`.

   A python script for training of different feature extractor / style image combinations can be found in `train_multiple.py`.

   
<p align="right">(<a href="#top">back to top</a>)</p>

### Test

Download the pretrained [ControlNet Model](https://huggingface.co/lllyasviel/sd-controlnet-canny) and save it at `./models/sd/control_sd15_canny.pth`.

   To generate new images, configure a TestConfig under `config/test/current.py` and run `python inference.py`.
   
   To use controlled image generation, `controlled=True` has to be set in the TestConfig, the ControlNet Stable Diffusion checkpoint has to be set `sd_checkpoint='models/sd/control_sd15_canny.pth'` and the correct model config has to be loaded `model_config='configs/stable-diffusion/v1-controlled-inference.yaml'`.
   
<p align="right">(<a href="#top">back to top</a>)</p>

### Data

The gathered set of CC0 content images can be found in [Google Drive](https://drive.google.com/drive/folders/1BfK9FJYw8GjKDI1jalKz-ZXriVC0VNpM?usp=drive_link).

<p align="right">(<a href="#top">back to top</a>)</p>

### Citation
   
   ```
   @masterthesis{heid2024reducing,
    author    = {Rafael Heid},
    title     = {Reducing object distortion in inversion-based image style transfer},
    school    = {University of Hamburg},
    month     = {September},
    year      = {2024},
    type      = {Master's thesis}
}
   ```
   
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- 
<!-- USAGE EXAMPLES -->
<!-- ## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- ROADMAP -->
<!-- ## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
<!-- ## Contributing -->

<!-- Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
 -->
<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->




<!-- LICENSE -->
<!-- ## License -->
<!-- 
Distributed under the MIT License. See `LICENSE.txt` for more information.
 -->
<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- CONTACT -->


<!-- 
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
 -->



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments -->
<!-- 
Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search) -->

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
