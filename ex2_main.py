from ex2_utils import *

import matplotlib.pyplot as plt
import time


def conv1Demo(inSignal: np.ndarray, kernel: np.ndarray):
    start = time.time()
    conv1 = conv1D(inSignal, kernel)

    print("conv1D time:%.2f" % (time.time() - start))
    kernel_str = "[ "
    for i in range(len(kernel)):
        kernel_str += str(kernel[i])
        kernel_str += " "
    kernel_str += "] "
    # print("inSignal:", inSignal)
    print(inSignal, "*", kernel_str, "=")
    print(conv1)
    print("open cv ans:", np.convolve(inSignal, kernel, 'full'), "\n")


def conv2Demo(img: np.ndarray, kernel: np.ndarray):
    start = time.time()
    conv2_img = conv2D(img, kernel)
    cv_ans = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    print("conv2D time:%.2f" % (time.time() - start))

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image", fontdict=None, loc=None, pad=20, y=None)
    ax[1].imshow(conv2_img, cmap='gray')
    ax[2].imshow(cv_ans,  cmap='gray')
    ax[2].set_title("cv_ans", fontdict=None, loc=None, pad=20, y=None)

    label = "after Convolution with\n[ "
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            label += str(kernel[i][j])[0: 4]
            label += " "
        if i != len(kernel)-1:
            label += "\n"
    label += "]"
    ax[1].set_title(label, fontdict=None, loc=None, pad=0.1, y=None)
    ax[3].imshow(cv_ans-conv2_img,  cmap='gray')
    ax[3].set_title("diffrance: cv-my", fontdict=None, loc=None, pad=20, y=None)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    plt.savefig("conv2D", dpi=300, bbox_inches='tight')

    plt.show()


def derivDemo(img: np.ndarray):
    start = time.time()
    direction, magnitude, Ix, Ix = convDerivative(img)
    print("convDerivative time:%.2f" % (time.time() - start))

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(direction, cmap='gray')
    ax[0][0].set_title("direction", fontdict=None, loc=None, pad=8, y=None)
    ax[0][1].imshow(magnitude, cmap='gray')
    ax[0][1].set_title("magnitude", fontdict=None, loc=None, pad=8, y=None)
    ax[1][0].imshow(Ix, cmap='gray')
    ax[1][0].set_title("x derivative", fontdict=None, loc=None, pad=8, y=None)
    ax[1][1].imshow(Ix, cmap='gray')
    ax[1][1].set_title("y derivative", fontdict=None, loc=None, pad=8, y=None)
    plt.savefig("derive", dpi=300, bbox_inches='tight')
    plt.show()


def blurDemo(in_img: np.ndarray, kernel_size: np.ndarray):
    start = time.time()
    my_output = blurImage1(in_img, kernel_size)
    cv_output = blurImage2(in_img, kernel_size)
    print("blurImage1 and blurImage2 time:%.2f" % (time.time() - start))

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(in_img, cmap='gray')
    ax[1].imshow(my_output, cmap='gray')
    ax[2].imshow(cv_output, cmap='gray')
    ax[0].set_title("original image", fontdict=None, loc=None, pad=10, y=None)
    ax[1].set_title("with my \ngaussian filter", fontdict=None, loc=None, pad=10, y=None)
    ax[2].set_title("with cv \ngaussian filter", fontdict=None, loc=None, pad=10, y=None)
    plt.savefig("blur", dpi=300, bbox_inches='tight')
    plt.show()


def edgeDemo(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray):
    # sobel:
    start = time.time()
    cv_ans, out_img = edgeDetectionSobel(img1, 0.5)
    print("edgeDetectionSobel time:%.2f" % (time.time() - start))
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(out_img, cmap='gray')
    ax[2].imshow(cv_ans, cmap='gray')
    ax[0].set_title("image\nbefore sobel", fontdict=None, loc=None, pad=20, y=None)
    ax[1].set_title("my solution", fontdict=None, loc=None, pad=20, y=None)
    ax[2].set_title("opencv solution", fontdict=None, loc=None, pad=20, y=None)
    plt.savefig("edge_sobel", dpi=300, bbox_inches='tight')
    plt.show()

    # zero crossing LOG :
    start = time.time()
    edge_img = edgeDetectionZeroCrossingLOG(img2)
    print("edgeDetectionZeroCrossingLOG time:%.2f" % (time.time() - start))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img2, 'gray')
    ax[0].set_title("original image", fontdict=None, loc=None, pad=20, y=None)
    ax[1].imshow(edge_img, 'gray')
    ax[1].set_title(" after zero crossing", fontdict=None, loc=None, pad=20, y=None)
    plt.savefig("edge_LOG", dpi=300, bbox_inches='tight')

    plt.show()

    # canny:
    thrs_1 = 200
    thrs_2 = 100
    start = time.time()
    cv_canny, my_canny = edgeDetectionCanny(img3, thrs_1, thrs_2)
    print("edgeDetectionCanny time:%.2f" % (time.time() - start))
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img3, cmap='gray')
    ax[1].imshow(my_canny, cmap='gray')
    ax[2].imshow(cv_canny, cmap='gray')
    ax[0].set_title("image \nbefore Canny", fontdict=None, loc=None, pad=20, y=None)
    ax[1].set_title("my solution", fontdict=None, loc=None, pad=20, y=None)
    ax[2].set_title("opencv solution", fontdict=None, loc=None, pad=20, y=None)
    plt.savefig("edge_canny", dpi=300, bbox_inches='tight')
    plt.show()


def houghDemo(img: np.ndarray, min_radius: float, max_radius: float):
    start = time.time()
    ans = houghCircle(img, min_radius, max_radius)
    print("houghCircle time:%.2f" % (time.time() - start))
    drew_circles(img, ans)
    print("ans:", ans)


def drew_circles(img: np.ndarray, ans: list):
    fig, ax = plt.subplots()
    for x, y, r in ans:
        circle = plt.Circle((x, y), r, color='red', fill=False)
        center = plt.Circle((x, y), 0.5, color='red')
        ax.add_patch(circle)
        ax.add_patch(center)
    ax.imshow(img)
    plt.title("hough circle")
    plt.savefig("houghCircle", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("ID:", 316333632)
    print()
    # images:
    beach = cv2.cvtColor(cv2.imread("beach.jpg"), cv2.COLOR_BGR2GRAY)
    img_balls = cv2.cvtColor(cv2.imread("pool_balls.jpeg"), cv2.COLOR_BGR2GRAY)
    boxman = cv2.cvtColor(cv2.imread("boxman.jpg"), cv2.COLOR_BGR2GRAY)
    monkey = cv2.cvtColor(cv2.imread("codeMonkey.jpeg"), cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2GRAY)
    coins = cv2.cvtColor(cv2.imread("coins.jpg"), cv2.COLOR_BGR2GRAY)

    # Convolution 1D:
    inSignal1 = np.array([1, 2, 3, 4, 5])
    kernel1D_1 = np.array([0, 1, 0])
    kernel1D_2 = np.array([0, 2, 0])
    kernel1D_3 = np.array([1, 2, 3])
    kernel1D_4 = np.array([1/3, 1/3, 1/3])
    conv1Demo(inSignal1, kernel1D_1)
    conv1Demo(inSignal1, kernel1D_2)
    conv1Demo(inSignal1, kernel1D_3)
    conv1Demo(inSignal1, kernel1D_4)
    inSignal2 = np.array([1, 2, 3])
    kernel = np.array([1, 1])
    conv1Demo(inSignal2, kernel)

    # Convolution 2D:
    kernel2D_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel2D_2 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    kernel2D_3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    conv2Demo(boxman, kernel2D_1)
    conv2Demo(beach, kernel2D_2)
    conv2Demo(monkey, kernel2D_3)

    derivDemo(beach)
    derivDemo(boxman)

    blurDemo(img1, (5, 5))

    edgeDemo(coins, beach, boxman)

    houghDemo(img_balls, 18, 20)
    houghDemo(coins, 60, 100)


if __name__ == '__main__':
    main()


