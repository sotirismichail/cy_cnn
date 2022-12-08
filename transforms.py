import numpy as np


def _lpcoords(ishape, w, angles=None):
    ishape = np.array(ishape)
    bands = ishape[2]

    oshape = ishape.copy()
    centre = (ishape[:2]-1)/2.

    d = np.hypot(*(ishape[:2]-centre))
    log_base = np.log(d)/w

    if angles is None:
        angles = -np.linspace(0, 2*np.pi, 2*w+1)[:-1]
    theta = np.empty((len(angles), w), dtype=np.float64) # todo: add a global config file, define the ftype there
    theta.transpose()[:] = angles
    L = np.empty_like(theta)
    L[:] = np.arange(w).astype(np.float64)

    r = np.exp(L*log_base)

    return r*np.sin(theta) + centre[0], r*np.cos(theta) + centre[1], angles, log_base

'''
  polar_cart

  Arguments:
    Arg1: r -> radial distance to origin
    Arg2: theta -> azimuthal angle (with respect to polar axis)
    Arg3: centre -> polar axis centre

  Returns:
    kernel_list -> list of kernels to be used for convolution

  Description: Calculating the reverse polar angle, used for the logpolar transform
'''


def polar_cart(r, theta, centre):
    x = r * np.cos(theta) + centre[0]
    y = r * np.sin(theta) + centre[1]

    return x, y


def img_polar(img, centre, final_radius, initial_radius=0, phase_width=3000):

    theta, radius = np.meshgrid(np.linspace(0, 2*np.pi, phase_width),
                                np.arange(initial_radius, final_radius))

    x_cart, y_cart = polar_cart(radius, theta, centre)
    x_cart = x_cart.astype(int)
    y_cart = y_cart.astype(int)

    polar_img = img[y_cart, x_cart, :]
    polar_img = np.reshape(polar_img, (final_radius - initial_radius, phase_width, 3))

    return polar_img
