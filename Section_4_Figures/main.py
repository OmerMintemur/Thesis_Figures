import numpy as np
import cv2
import matplotlib.pyplot as plt

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
def RoseCurveButterworth(image, a, k, beta, W, order):
    x = 0
    y = 0
    matrix = np.zeros((image.shape[0], image.shape[1]))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            x = float(r - matrix.shape[0] / 2)
            y = float(c - matrix.shape[1] / 2)
            theta = findangle(x, y)
            rad = np.sqrt(a * a * np.cos(k * theta * (np.pi / 180) + 2 * beta * (np.pi / 180)))
            d = np.sqrt(np.power((r - matrix.shape[0] / 2), 2) + np.power((c - matrix.shape[0] / 2), 2))
            if d < rad:
                D = np.sqrt(np.power((r - matrix.shape[0] / 2), 2) + np.power((c - matrix.shape[0] / 2), 2))
                sq_d = np.power(rad, 2)
                sq_c = np.power(d, 2)
                den = sq_d - sq_c
                nom = D * W
                matrix[r, c] = (1 / (1 + np.power((nom / den), 2 * order)))
    return matrix

def RoseCurveNormal(image, a, k, beta):
    x = 0
    y = 0
    matrix = np.zeros((image.shape[0], image.shape[1]))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            x = float(r - matrix.shape[0] / 2)
            y = float(c - matrix.shape[1] / 2)
            theta = findangle(x, y)
            rad = np.sqrt(a * a * np.cos(k * theta * (np.pi / 180) + 2 * beta * (np.pi / 180)))
            d = np.sqrt(x * x + y * y)
            if d < rad:
                matrix[r][c] = 1
    return matrix


def RoseCurveGaussian(image, a, k, beta, W):
    x = 0
    y = 0
    matrix = np.zeros((image.shape[0], image.shape[1]))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            x = float(r - matrix.shape[0] / 2)
            y = float(c - matrix.shape[1] / 2)
            theta = findangle(x, y)
            rad = np.sqrt(a * a * np.cos(k * theta * (np.pi / 180) + 2 * beta * (np.pi / 180)))
            d = np.sqrt(np.power((r - matrix.shape[0] / 2), 2) + np.power((c - matrix.shape[0] / 2), 2))
            if d < rad:
                D = np.sqrt(np.power((r - matrix.shape[0] / 2), 2) + np.power((c - matrix.shape[0] / 2), 2))
                sq_d = np.power(rad, 2)
                sq_c = np.power(d, 2)
                den = sq_d - sq_c
                nom = D * W
                matrix[r, c] = 1 - np.exp(-np.power((den / nom), 2))
    return matrix

def lemniscateFilter(img, high_pass_type, high_pass_radius, angle,order):
    if high_pass_type == "GAUSSIAN":
        fft_of_image = np.fft.fftshift(np.fft.fft2(img))
        gaussian_hp = gaussianHighPass(img, high_pass_radius)
        gaussian_lemniscate = RoseCurveGaussian(img, 512 / 2, 2, angle, high_pass_radius)
        final = np.multiply(np.multiply(fft_of_image, gaussian_hp), gaussian_lemniscate)
        result = np.fft.ifft2(final)
        result = abs(result)
        result /= (result.max() / 255.0)
        result = result.astype(int)
        return result, gaussian_lemniscate, gaussian_hp
    if high_pass_type=="BUTTERWORTH":
        fft_of_image = np.fft.fftshift(np.fft.fft2(img))
        butterworth_hp = butterworthHighPass(img, high_pass_radius, order)
        butterworth_lemniscate = RoseCurveButterworth(img, 512 / 2, 2, angle, high_pass_radius,order)
        final = np.multiply(np.multiply(fft_of_image, butterworth_hp), butterworth_lemniscate)
        result = np.fft.ifft2(final)
        result = abs(result)
        result /= (result.max() / 255.0)
        result = result.astype(int)
        return result, butterworth_lemniscate, butterworth_hp

def gaussianHighPass(matrix, radius):
    matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    center = (matrix.shape[0] / 2, matrix.shape[0] / 2)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            dist = np.sqrt(np.power((r - matrix.shape[0] / 2), 2) + np.power((c - matrix.shape[0] / 2), 2))
            d = distance((r, c), center)
            matrix[r, c] = 1 - np.exp(-(np.power(dist, 2)) / (2 * (radius ** 2)))
    return matrix
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def butterworthHighPass(matrix, radius, order):
    matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    center = (matrix.shape[0] / 2, matrix.shape[0] / 2)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            dist = np.sqrt(np.power((r - matrix.shape[0] / 2), 2) + np.power((c - matrix.shape[0] / 2), 2))
            matrix[r, c] = 1 - 1 / (1 + (distance((r, c), center) / radius) ** (2 * order))
    return matrix


#FIGURE 4.6 (a-b) BEGIN
# lena_image = cv2.imread("lena512.bmp", 0)
# lena_image_fourier = np.fft.fftshift(np.fft.fft2(lena_image))
# f = RoseCurveNormal(lena_image, 256, 2, 90)
# final = np.multiply(f, lena_image_fourier)
# result = np.fft.ifft2(final)
# result = abs(result)
# result /= (result.max() / 255.0)
# result = result.astype(int)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
#
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
#
# plt.xlabel('X', fontsize=8, labelpad=10)
# plt.ylabel('Y', fontsize=8, rotation=0, labelpad=10)
# # plt.axis('off')
# plt.imshow(result, cmap="gray")
#
# plt.show()
# fig.savefig("Figure_4_6_b.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
# #FIGURE 4.6 (a-b) END


#FIGURE 4.7 (a-b) BEGIN

# high_pass_filter = np.zeros((512, 512), dtype=float)
# for r in range(high_pass_filter.shape[0]):
#     for c in range(high_pass_filter.shape[1]):
#         d_u_v = np.sqrt(np.power(r-(high_pass_filter.shape[0]/2), 2) + np.power(c-(high_pass_filter.shape[1]/2), 2))
#         if d_u_v > 60:
#             high_pass_filter[r, c] = 1
#         else:
#             high_pass_filter[r, c] = 0
# lena_image = cv2.imread("lena512.bmp", 0)
# lena_image_fourier = np.fft.fftshift(np.fft.fft2(lena_image))
# f = RoseCurveNormal(lena_image, 256, 2, 90)
# final = np.multiply(f,np.multiply(high_pass_filter,lena_image_fourier))
# result = np.fft.ifft2(final)
# result = abs(result)
# result /= (result.max() / 255.0)
# result = result.astype(int)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
#
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
#
# plt.xlabel('X', fontsize=8, labelpad=10)
# plt.ylabel('Y', fontsize=8, rotation=0, labelpad=10)
# # plt.axis('off')
# plt.imshow(result, cmap="gray")
#
# plt.show()
# fig.savefig("Figure_4_.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#FIGURE 4.7 (a-b) END


#FIGURE 4.8 BEGIN
# high_pass_filter = np.zeros((512, 512), dtype=float)
# for r in range(high_pass_filter.shape[0]):
#     for c in range(high_pass_filter.shape[1]):
#         d_u_v = np.sqrt(np.power(r-(high_pass_filter.shape[0]/2), 2) + np.power(c-(high_pass_filter.shape[1]/2), 2))
#         if d_u_v > 120:
#             high_pass_filter[r, c] = 1
#         else:
#             high_pass_filter[r, c] = 0
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
#
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
#
# # plt.xlabel('U', fontsize=8, labelpad=10)
# # plt.ylabel('V', fontsize=8, rotation=0, labelpad=10)
# # plt.axis('off')
# plt.imshow(high_pass_filter, cmap="gray")
#
# plt.show()
# fig.savefig("Figure_4_8_d.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#FIGURE 4.8 END

#FIGURE 4.9 BEGIN
# high_pass_filter = np.zeros((512, 512), dtype=float)
# for r in range(high_pass_filter.shape[0]):
#     for c in range(high_pass_filter.shape[1]):
#         d_u_v = np.sqrt(np.power(r-(high_pass_filter.shape[0]/2), 2) + np.power(c-(high_pass_filter.shape[1]/2), 2))
#         if d_u_v > 60:
#             high_pass_filter[r, c] = 1
#         else:
#             high_pass_filter[r, c] = 0
# circle = np.zeros((512, 512), dtype=float)
# for r in range(circle.shape[0]):
#     for c in range(circle.shape[1]):
#         d_u_v = np.sqrt(np.power(r-(circle.shape[0]/2), 2) + np.power(c-(circle.shape[1]/2), 2))
#         if d_u_v > 120:
#             circle[r, c] = 1
#         else:
#             circle[r, c] = 0
# circle_fourier = np.fft.fftshift(np.fft.fft2(circle))
# for x in range(0, 180, 15):
#     print(x)
#     f = RoseCurveNormal(circle, 256, 2, x)
#     final = np.multiply(f, np.multiply(high_pass_filter,circle_fourier))
#     result = np.fft.ifft2(final)
#     result = abs(result)
#     result /= (result.max() / 255.0)
#     result = result.astype(int)
#     image_name = input("Enter an Image Name for Figure 4.9 (a, b, c, d, e, f, g, h, i, j, k, l)")
#     image_name = "Figure_4_9_"+str(image_name)
#     print(image_name)
#     cm = 1 / 2.54
#     xtick = [0, 200, 400]
#     ytick = [0, 200, 400]
#     fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
#     # plt.xticks(xtick, fontsize=8)
#     # plt.yticks(ytick, fontsize=8)
#     plt.axis('off')
#     plt.imshow(result, cmap="gray")
#     # plt.xlabel(coordx, fontsize=8)
#     # plt.ylabel(coordy, fontsize=8, rotation=0)
#     plt.show()
#     fig.savefig(str(image_name) + ".png", format="png", dpi=300, bbox_inches='tight', transparent=True)

#FIGURE 4.9 END

#FIGURE 4.10 BEGIN

# lena_image = cv2.imread("Figure_4_10.png", 0)
# import matplotlib.patches as patches
#
# # Create a Rectangle patch
# rect = patches.Rectangle((470, 150), 100, 400, linewidth=2, edgecolor='r',facecolor='none')
# cm = 1 / 2.54
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# # Add the patch to the Axes
# ax.add_patch(rect)
# plt.imshow(lena_image, cmap="gray")
# plt.axis('off')
# plt.show()
#
# fig.savefig("Figure_4_10_a.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
# y=150
# x=470
# h=400
# w=100
# crop = lena_image[y:y+h, x:x+w]
# cm = 1 / 2.54
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.imshow(crop, cmap="gray")
# plt.axis('off')
# plt.show()
# fig.savefig("Figure_4_10_b.png", format="png", dpi=300, bbox_inches='tight', transparent=True)

#FIGURE 4.10 END

#FIGURE 4.11 BEGIN
# lena_image = cv2.imread("lena512.bmp", 0)
# lena_image_fourier = np.fft.fftshift(np.fft.fft2(lena_image))
# result, gaussian_lemniscate, gaussian_hp = lemniscateFilter(lena_image,"GAUSSIAN", 60, 90)
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(gaussian_hp, cmap="gray")
# plt.xlabel("U", fontsize=8)
# plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_4_11_b.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(gaussian_lemniscate, cmap="gray")
# plt.xlabel("U", fontsize=8)
# plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_4_11_c.png", format="png", dpi=300, bbox_inches='tight', transparent=True)

# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(6.8*cm, 7.04*cm))
# # plt.xticks(xtick, fontsize=8)
# # plt.yticks(ytick, fontsize=8)
# plt.axis('off')
# plt.imshow(result, cmap="gray")
# # plt.xlabel(coordx, fontsize=8)
# # plt.ylabel(coordy, fontsize=8, rotation=0)
# plt.show()
# fig.savefig("Figure_4_12_b.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#FIGURE 4.11 END

#FIGURE 4.12 BEGIN
# lena_image = cv2.imread("Figure_4_12_b.png", 0)
# import matplotlib.patches as patches
#
# # Create a Rectangle patch
# rect = patches.Rectangle((470, 150), 100, 400, linewidth=2, edgecolor='r',facecolor='none')
# cm = 1 / 2.54
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# # Add the patch to the Axes
# ax.add_patch(rect)
# plt.imshow(lena_image, cmap="gray")
# plt.axis('off')
# plt.show()
#
# fig.savefig("Figure_4_12_b_1.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
# y=150
# x=470
# h=400
# w=100
# crop = lena_image[y:y+h, x:x+w]
# cm = 1 / 2.54
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.imshow(crop, cmap="gray")
# plt.axis('off')
# plt.show()
# fig.savefig("Figure_4_12_d.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#FIGURE 4.12 END

#FIGURE 4.13 BEGIN
# circle = np.zeros((512, 512), dtype=float)
# for r in range(circle.shape[0]):
#     for c in range(circle.shape[1]):
#         d_u_v = np.sqrt(np.power(r-(circle.shape[0]/2), 2) + np.power(c-(circle.shape[1]/2), 2))
#         if d_u_v > 120:
#             circle[r, c] = 1
#         else:
#             circle[r, c] = 0
# circle_fourier = np.fft.fftshift(np.fft.fft2(circle))
# for x in range(0, 180, 15):
#     print(x)
#     result, gaussian_lemniscate, gaussian_hp = lemniscateFilter(circle, "GAUSSIAN", 60, x)
#     image_name = input("Enter an Image Name for Figure 4.13 (a, b, c, d, e, f, g, h, i, j, k, l)")
#     image_name = "Figure_4_13_"+str(image_name)
#     print(image_name)
#     cm = 1 / 2.54
#     xtick = [0, 200, 400]
#     ytick = [0, 200, 400]
#     fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
#     # plt.xticks(xtick, fontsize=8)
#     # plt.yticks(ytick, fontsize=8)
#     plt.axis('off')
#     plt.imshow(result, cmap="gray")
#     # plt.xlabel(coordx, fontsize=8)
#     # plt.ylabel(coordy, fontsize=8, rotation=0)
#     plt.show()
#     fig.savefig(str(image_name) + ".png", format="png", dpi=300, bbox_inches='tight', transparent=True)

#FIGURE 4.13 END

#FIGURE 4.14 BEGIN
# lena_image = cv2.imread("lena512.bmp", 0)
# lena_image_fourier = np.fft.fftshift(np.fft.fft2(lena_image))
# result, gaussian_lemniscate, gaussian_hp = lemniscateFilter(lena_image,"BUTTERWORTH", 60, 90, 2)
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(gaussian_hp, cmap="gray")
# plt.xlabel("U", fontsize=8)
# plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_4_14_b.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(gaussian_lemniscate, cmap="gray")
# plt.xlabel("U", fontsize=8)
# plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_4_14_c.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
# cm = 1 / 2.54
# xtick = [0, 200, 400]
# ytick = [0, 200, 400]
# fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
# plt.xticks([], fontsize=8)
# plt.yticks([], fontsize=8)
# # plt.axis('off')
# plt.imshow(result, cmap="gray")
# # plt.xlabel("U", fontsize=8)
# # plt.ylabel("V", fontsize=8, rotation=0,labelpad=10)
# plt.show()
# fig.savefig("Figure_4_14_d.png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#FIGURE 4.14 END

#FIGURE 4.15 BEGIN
circle = np.zeros((512, 512), dtype=float)
for r in range(circle.shape[0]):
    for c in range(circle.shape[1]):
        d_u_v = np.sqrt(np.power(r-(circle.shape[0]/2), 2) + np.power(c-(circle.shape[1]/2), 2))
        if d_u_v > 120:
            circle[r, c] = 1
        else:
            circle[r, c] = 0
circle_fourier = np.fft.fftshift(np.fft.fft2(circle))
for x in range(0, 180, 15):
    print(x)
    result, gaussian_lemniscate, gaussian_hp = lemniscateFilter(circle, "BUTTERWORTH", 60, x ,2)
    image_name = input("Enter an Image Name for Figure 4.15 (a, b, c, d, e, f, g, h, i, j, k, l)")
    image_name = "Figure_4_15_"+str(image_name)
    print(image_name)
    cm = 1 / 2.54
    xtick = [0, 200, 400]
    ytick = [0, 200, 400]
    fig, ax = plt.subplots(figsize=(5.5 * cm, 6.5 * cm))
    # plt.xticks(xtick, fontsize=8)
    # plt.yticks(ytick, fontsize=8)
    plt.axis('off')
    plt.imshow(result, cmap="gray")
    # plt.xlabel(coordx, fontsize=8)
    # plt.ylabel(coordy, fontsize=8, rotation=0)
    plt.show()
    fig.savefig(str(image_name) + ".png", format="png", dpi=300, bbox_inches='tight', transparent=True)
#FIGURE 4.15 END