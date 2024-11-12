"""
   Computes the Underwater Image Quality Measure (UIQM)
   metrics paper: https://ieeexplore.ieee.org/document/7305804
   referenced from  https://github.com/xahidbuffon/FUnIE-GAN/blob/master/Evaluation/uqim_utils.py
"""
from scipy import ndimage
import numpy as np
import math


def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    # calculate mu_alpha weight
    weight = (1 / (K - T_a_L - T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val


# Calculates the squared average deviation of each pixel value in the input list x from the mean mu.
def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)


def _uicm(x):
    # Extract the Red, Green, and Blue channels and flatten them
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    # Compute the difference between the Red and Green channels (RG component)
    RG = R - G
    # Compute the difference between the average of Red and Green channels and the Blue channel (YB component)
    YB = ((R + G) / 2) - B
    # Compute the mean of the RG and YB components using the function mu_a()
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    # Compute the standard deviation of the RG and YB components using the function s_a()
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    # Compute the magnitude of the chroma feature (l) using Euclidean distance
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    # Compute the overall contrast feature (r) as the square root of the sum of variances
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def sobel(x):
    # Compute the Sobel gradient in the x-direction
    dx = ndimage.sobel(x, 0)
    # Compute the Sobel gradient in the y-direction
    dy = ndimage.sobel(x, 1)
    # Compute the magnitude of the gradient
    mag = np.hypot(dx, dy)
    # Scale the magnitude to the range [0, 255]
    mag *= 255.0 / np.max(mag)
    return mag


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # Calculate the number of blocks in the horizontal and vertical directions
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # Calculate the weight for normalization
    w = 2. / (k1 * k2)

    # Define the block size
    blocksize_x = window_size
    blocksize_y = window_size

    # Ensure that the image dimensions are divisible by the window size
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    # Initialize the value variable
    val = 0
    # Iterate over each block
    for l in range(k1):
        for k in range(k2):
            # Extract the current block
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            # Calculate the maximum and minimum values in the block
            max_ = np.max(block)
            min_ = np.min(block)

            # bound checks, can't do log(0) ,   # Check bounds to avoid log(0)
            if min_ == 0.0:
                val += 0
            elif max_ == 0.0:
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]

    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    # get eme for each channel,
    r_eme = eme(R_edge_map, 8)
    g_eme = eme(G_edge_map, 8)
    b_eme = eme(B_edge_map, 8)

    # Define coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    # Calculate weighted sum of the values
    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def plip_g(x, mu=1026.0):
    return mu - x


def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))


def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / (gamma))


def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
    """
    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0

    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = -1. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    # entropy scale - higher helps with randomness
    alpha = 1

    val = 0
    # Iterate over each block
    for l in range(k1):
        for k in range(k2):
            # Extract the current block
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            # Calculate the maximum and minimum values in the block
            max_ = np.max(block)
            min_ = np.min(block)
            # Calculate the difference and sum of max and min values
            top = max_ - min_
            bot = max_ + min_
            # Check for potential division by zero or NaN values
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                # Calculate the desired value based on certain conditions
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)
    return w * val


def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    # calculate contrast , color and sharpness
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm
