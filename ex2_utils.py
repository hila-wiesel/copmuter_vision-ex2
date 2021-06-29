import cv2
import numpy as np
from math import pi, cos, sin


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    flip_kernel = kernel1[::-1]
    kernel_len = len(kernel1)
    signal_longer = inSignal

    # padding with zeros the inSignal arr:
    for i in range(kernel_len - 1):
        signal_longer = np.insert(signal_longer, 0, 0)
        signal_longer = np.append(signal_longer, 0)

    new_img = np.zeros(len(inSignal) + kernel_len - 1)
    for i in range(kernel_len + len(inSignal) - 1):
        new_img[i] = ((flip_kernel * signal_longer[i:kernel_len + i]).sum())

    return new_img


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    flip_kernel = np.flipud(np.fliplr(kernel2))
    kernel_row = flip_kernel.shape[0]
    kernel_col = flip_kernel.shape[1]

    new_img = np.zeros_like(inImage)
    padded_img = padded_replicate(inImage, kernel_row, kernel_col)

    for x in range(inImage.shape[0]):
        for y in range(inImage.shape[1]):
            new_img[x, y] = (padded_img[x: x + kernel_col, y: y + kernel_row] * flip_kernel).sum()
            if flip_kernel.sum() != 0:
                new_img[x, y] /= flip_kernel.sum()

    return new_img


def padded_replicate(inImage: np.ndarray, kernel_row: int, kernel_col: int) -> np.ndarray:
    add_row = int((kernel_row - 1) / 2)
    add_col = int((kernel_col - 1) / 2)

    padded_img = np.zeros((inImage.shape[0] + kernel_row - 1, inImage.shape[1] + kernel_col - 1))
    padded_img[kernel_row - 2:-(kernel_row - 2), kernel_col - 2:-(kernel_col - 2)] = inImage

    up_row = padded_img[add_row]
    down_row = padded_img[-add_row - 1]
    padded_img_flip = np.flipud(np.fliplr(padded_img))
    left_col = padded_img_flip[add_col]
    right_col = padded_img_flip[-add_col - 1]

    for i in range(add_row):
        padded_img[i] = up_row
        padded_img[-i] = down_row
    for j in range(add_col):
        for row in range(padded_img.shape[0]):
            padded_img[row, j] = left_col[row]
            padded_img[row, -j] = right_col[-row]

    return padded_img


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude, x_der, y_der)
    """
    kernel_x = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    kernel_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

    # derivative by rows:
    Ix = cv2.filter2D(inImage, -1, kernel_x)  # , borderType=cv2.BORDER_REPLICATE

    # derivative by columns:
    Iy = cv2.filter2D(inImage, -1, kernel_y)

    eps = 0.0000000001
    magnitude = pow(Ix ** 2 + Iy ** 2, 0.5)
    direction = np.arctan(Iy / (Ix + eps))

    return direction, magnitude, Ix, Iy


# bonus:
def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    size = kernel_size[0]
    sigma = 1
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    in_image = cv2.filter2D(in_image, -1, g)
    return in_image


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma=0)
    out_img = cv2.filter2D(in_image, -1, gaussian_kernel)
    return out_img


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    thresh = thresh * 255

    smooth_x = cv2.filter2D(img, -1, Gx, borderType=cv2.BORDER_REPLICATE).astype(float)
    smooth_y = cv2.filter2D(img, -1, Gy, borderType=cv2.BORDER_REPLICATE).astype(float)

    my_sobel_image = np.sqrt(smooth_x ** 2 + smooth_y ** 2)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    cv_ans = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    change_by_thresh(my_sobel_image, thresh)
    change_by_thresh(cv_ans, thresh)

    return cv_ans, my_sobel_image


def change_by_thresh(img: np.ndarray, thresh: float = 0.7):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] >= thresh:
                img[x, y] = 255
            else:
                img[x, y] = 0


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """

    # smooth the image with a Gaussian filter:
    img = img.astype(np.float)
    smooth_img = cv2.GaussianBlur(img, (3, 3), 0)

    # convolve the smoothed image with the Laplacian filter:
    kernel_laplacian = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    derive2_img = cv2.filter2D(smooth_img, -1, kernel=kernel_laplacian,
                               borderType=cv2.BORDER_REPLICATE)  # ddepth=cv2.CV_16S,

    # find the following patterns {-,0,+} or {-,+} :
    edge_img = np.zeros_like(derive2_img)
    row, col = img.shape
    for x in range(1, row - 1):
        for y in range(1, col - 1):
            wind = derive2_img[x - 1:x + 2, y - 1:y + 2]
            wind_max = wind.max()
            wind_min = wind.min()
            pixel = derive2_img[x, y]
            if pixel == 0.0 and wind_min <= 0.0 <= wind_max != wind_min:
                edge_img[x, y] = 255
            elif (wind_min <= 0.0 <= pixel != wind_min) or (pixel < 0.0 < wind_max != pixel):
                edge_img[x, y] = 255
            else:
                edge_img[x, y] = 0

    return edge_img


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    # smooth the image with a Gaussian filter:
    smooth_img = cv2.GaussianBlur(img, (3, 3), 0)

    # Compute the partial derivatives Ix, Iy, and magnitude and direction of the gradient:
    directions, magnitude, Ix, Iy = convDerivative(smooth_img)
    angle = np.degrees(directions)
    magnitude = (magnitude / magnitude.max()) * 255

    # non maximum suppression:
    edge_img = np.zeros_like(magnitude)
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            ni1 = ni2 = magnitude[i, j]
            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                ni1 = magnitude[i, j + 1]
                ni2 = magnitude[i, j - 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                ni1 = magnitude[i + 1, j - 1]
                ni2 = magnitude[i - 1, j + 1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                ni1 = magnitude[i + 1, j]
                ni2 = magnitude[i - 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                ni1 = magnitude[i - 1, j - 1]
                ni2 = magnitude[i + 1, j + 1]
            if (magnitude[i, j] >= ni1) and (magnitude[i, j] >= ni2):
                edge_img[i, j] = magnitude[i, j]
            else:
                edge_img[i, j] = 0

    # hysteresis:
    strong_edges = np.zeros_like(edge_img)
    for x in range(M):
        for y in range(N):
            if edge_img[x, y] > thrs_1:
                strong_edges[x, y] = edge_img[x, y]

    for x in range(M):
        for y in range(N):
            if edge_img[x, y] <= thrs_2:
                edge_img[x, y] = 0
            if thrs_2 < edge_img[x, y] <= thrs_1:
                if strong_edges[x - 1, y] == strong_edges[x + 1, y] == strong_edges[x, y - 1] == strong_edges[
                    x, y + 1] == strong_edges[x - 1, y - 1] == strong_edges[x + 1, y + 1] == strong_edges[
                    x + 1, y - 1] == strong_edges[x - 1, y + 1] == 0:
                    edge_img[x, y] = 0

    cv_ans = cv2.Canny(img, thrs_2, thrs_1)
    return cv_ans, edge_img


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    canny_cv, canny_my = edgeDetectionCanny(img, 200, 100)
    edges = []

    for x in range(canny_cv.shape[0]):
        for y in range(canny_cv.shape[1]):
            if canny_cv[x, y] == 255:
                edges.append((x, y))

    thresh = 0.47  # at least 47% of the pixels of a circle must be detected
    steps = 100  # number of samples from each circle

    points = []
    for r in range(min_radius, max_radius + 1):
        for t in range(steps):
            alpha = 2 * pi * t / steps
            x = int(r * cos(alpha))
            y = int(r * sin(alpha))
            points.append((x, y, r))

    temp_circles = {}  # dict{circle center, radius: counter}
    for x, y in edges:  # iterate the pixels of the edges:
        for dx, dy, r in points:
            b = x - dx
            a = y - dy
            count = temp_circles.get((a, b, r))
            if count is None:
                count = 0
            temp_circles[(a, b, r)] = count + 1

    # now add the appropriate circles to the ans list:
    circles = []
    sorted_temp = sorted(temp_circles.items(), key=lambda i: -i[1])
    for circle, counter in sorted_temp:
        x, y, r = circle
        # once a circle has been selected, we reject all the circles whose center is inside that circle
        if counter / steps >= thresh and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))

    return circles
