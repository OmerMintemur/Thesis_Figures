import numpy as np
import cv2
import matplotlib.pyplot as plt


# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

# g_kernel = cv2.getGaborKernel((512, 512), 8.0, np.pi, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#
# img = cv2.imread('test.jpg')
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
# #
# # cv2.imshow('image', img)
# # cv2.imshow('filtered image', filtered_img)
#
#
# g_kernel = cv2.resize(g_kernel, (512, 512), interpolation=cv2.INTER_CUBIC)
# fo = np.fft.fftshift(np.fft.fft2(g_kernel))
#
#
# #FIGURE 3.1 (a) BEGIN
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(g_kernel)
# plt.xlabel("X", fontsize=8)
# plt.ylabel("Y", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_3_1.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
# #FIGURE 3.1 (a) END
#
# #FIGURE 3.1 (b) BEGIN
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(np.abs(fo))
# plt.xlabel("U", fontsize=8)
# plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_3_2.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
# # #FIGURE 3.1 (b) END
#
# #FIGURE 3.2 BEGIN
# g_kernel_1 = cv2.getGaborKernel((512, 512), 8.0, np.pi, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel_2 = cv2.getGaborKernel((512, 512), 8.0, np.pi/2, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel_3 = cv2.getGaborKernel((512, 512), 8.0, np.pi/3, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel_4 = cv2.getGaborKernel((512, 512), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#
# # g_kernel = cv2.resize(g_kernel, (512, 512), interpolation=cv2.INTER_CUBIC)
# fo_1 = np.abs(np.fft.fftshift(np.fft.fft2(g_kernel_1)))
# fo_2 = np.abs(np.fft.fftshift(np.fft.fft2(g_kernel_2)))
# fo_3 = np.abs(np.fft.fftshift(np.fft.fft2(g_kernel_3)))
# fo_4 = np.abs(np.fft.fftshift(np.fft.fft2(g_kernel_4)))
# f = cv2.add(cv2.add(cv2.add(fo_1, fo_2), fo_3), fo_4)
#
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(6.8*cm, 7.04*cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(np.abs(f))
# plt.xlabel("U", fontsize=8)
# plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_3_3.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#
# #FIGURE 3.2 END

#FIGURE 3.3 BEGIN
# g_kernel_1 = cv2.getGaborKernel((32, 32), 8.0, np.pi, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel_2 = cv2.getGaborKernel((32, 32), 8.0, np.pi/2, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel_3 = cv2.getGaborKernel((32, 32), 8.0, np.pi/3, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel_4 = cv2.getGaborKernel((32, 32), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# img = cv2.imread('lena512.bmp')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# filtered_img_1 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_1)
# filtered_img_2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_2)
# filtered_img_3 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_3)
# filtered_img_4 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_4)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(4.5 * cm, 4.5 * cm))
# plt.axis('off')
# plt.imshow(filtered_img_1, cmap="gray")
# plt.show()
# fig.savefig("Figure_3_3(a).png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(4.5 * cm, 4.5 * cm))
# plt.axis('off')
# plt.imshow(filtered_img_2, cmap="gray")
# plt.show()
# fig.savefig("Figure_3_3(b).png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(4.5 * cm, 4.5 * cm))
# plt.axis('off')
# plt.imshow(filtered_img_3, cmap="gray")
# plt.show()
# fig.savefig("Figure_3_3(c).png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(4.5 * cm, 4.5 * cm))
# plt.axis('off')
# plt.imshow(filtered_img_4, cmap="gray")
# plt.show()
# fig.savefig("Figure_3_3(d).png", format="png", dpi=300, bbox_inches='tight', transparent=True)

#FIGURE 3.3 END

#FIGURE 3.7 BEGIN
def findangle(x, y):
    if x == 0:
        if y == 0:
            theta = 0.0
        elif y < 0:
            theta = 270.0
        else:
            theta = 90.0
    elif x < 0 and y == 0:
        theta = 180
    elif x < 0 and y > 0:
        x = -x
        theta = 180 - (np.arctan(float(y) / float(x)) * (180 / np.pi))
    elif x > 0 and y < 0:
        y = -y
        theta = 360 - (np.arctan(float(y) / float(x)) * (180 / np.pi))
    elif x < 0 and y < 0:
        y = -y
        x = -x
        theta = 180 + (np.arctan(float(y) / float(x)) * (180 / np.pi))
    else:
        theta = (np.arctan(float(y) / float(x)) * (180 / np.pi))
    return theta
def prepareFilter(row, column, select, total):
    matrix = np.zeros((row, column), dtype=np.uint8)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            x = float((r - int(matrix.shape[0] / 2)))
            y = float((c - int(matrix.shape[1] / 2)))
            theta = findangle(x, y)
            matrix = prepareMatrix(matrix, theta, r, c, select, total)
    return matrix


def prepareMatrix(matrix, theta, r, c, x, direction):
    x = (180 / direction * (x - 1) + 90) % 180
    if (theta > x and theta < x + 180 / direction) or (theta >= x + 180.0 and theta < 180.0 + x + 180 / direction):
        matrix[r][c] = 1
    return matrix


def apply_mask(matrix, mask_size):
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if abs(r - int(matrix.shape[0] / 2)) < mask_size and abs(c - int(matrix.shape[1] / 2)) < mask_size:
                matrix[r][c] = 0
    return matrix
def contourletFilter(img, direction, radius):
    filt_1 = prepareFilter(512, 512, direction, 8)
    filt_1 = apply_mask(filt_1, radius)
    fft_of_image = np.fft.fftshift(np.fft.fft2(img))
    final = np.multiply(fft_of_image, filt_1)
    result = np.fft.ifft2(final)
    result = abs(result)
    result /= (result.max() / 255.0)
    result = result.astype(int)
    return result, filt_1

img = cv2.imread("lena512.bmp", 0)
letters = ["b", "c", "d", "e", "f", "g", "h", "i"]
direc = [1, 2, 3, 4, 5, 6, 7,8]
for l, d in zip(letters, direc):
    print(str(l) + " " + str(d))
    process_image, filter_ = contourletFilter(img, d, 60)
    cm = 1 / 2.54
    xtick = [0, 200, 400]
    ytick = [0, 200, 400]
    fig, ax = plt.subplots(figsize=(4*cm, 4*cm))
    plt.xticks([], fontsize=8)
    plt.yticks([], fontsize=8)
    # plt.axis('off')
    plt.imshow(process_image, cmap="gray")
    # plt.xlabel("U", fontsize=8)
    # plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
    plt.show()
    fig.savefig("Figure_3_8_("+str(l)+").png", format="png", dpi=300, bbox_inches='tight', transparent=True)

#FIGURE 3.7 END