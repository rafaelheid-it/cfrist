from skimage.filters import difference_of_gaussians as _difference_of_gaussians
from skimage.filters import laplace as _laplace
from skimage.filters import hessian as _hessian
from skimage.feature import canny as _canny

def difference_of_gaussians(image, low_sigma=2, high_sigma=None, *, 
                                  mode='nearest', cval=0, channel_axis=None, 
                                  truncate=4.0):
    return _difference_of_gaussians(image, low_sigma, high_sigma, 
                                    mode=mode, cval=cval, channel_axis=channel_axis, truncate=truncate)

def laplace(image, ksize=3):
    return _laplace(image, ksize)

def hessian(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
            alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect',
            cval=0):
    return _hessian(image, sigmas, scale_range, scale_step, 
                    alpha, beta, gamma, black_ridges, mode, cval)

def canny(image, sigma=1., low_threshold=None, high_threshold=None,
          mask=None, use_quantiles=False, *, mode='constant', cval=0.0):
    return _canny(image, sigma, low_threshold, high_threshold, mask, use_quantiles, mode = mode, cval = cval)