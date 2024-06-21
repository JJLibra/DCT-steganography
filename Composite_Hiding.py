import cv2
import numpy as np
import pywt
from scipy.fftpack import fft, ifft


def xnor(a, b):
    return ~(a ^ b)


def embedding(cover_image, watermark_image1, alpha1, watermark_image2, alpha2, watermark_image3, alpha3):
    # 原始水印转化为灰度图
    def to_grayscale(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    watermark_image_gray1 = to_grayscale(watermark_image1)
    watermark_image_gray2 = to_grayscale(watermark_image2)
    watermark_image_gray3 = to_grayscale(watermark_image3)

    # 水印灰度图转化为二值图
    def to_binary(img_gray):
        return np.where(img_gray > 127, 1, 0).astype(np.float32)

    watermark_binary1 = to_binary(watermark_image_gray1)
    watermark_binary2 = to_binary(watermark_image_gray2)
    watermark_binary3 = to_binary(watermark_image_gray3)

    watermark_binary = watermark_binary1.astype(int) ^ watermark_binary2.astype(int) ^ watermark_binary3.astype(int)
    watermark = np.where(watermark_binary > 0.5, 255, 0).astype(np.uint8)

    # 原始图像转化为YUV图像
    cover_yuv = cv2.cvtColor(cover_image, cv2.COLOR_BGR2YUV)

    # 图像分块
    blocksize = 8
    height = cover_image.shape[0] // blocksize
    width = cover_image.shape[1] // blocksize

    # 对每个分块的YUV分量做DWT、DCT、SVD和DFT变换
    cover_image_blocks = np.zeros((height, width, blocksize, blocksize, 3), dtype=np.float32)
    embed_image_blocks = cover_image_blocks.copy()
    embed_image_yuv = np.zeros((cover_image.shape[0], cover_image.shape[1], 3))
    h_data = np.vsplit(cover_yuv, height)  # 图像垂直分割

    for h in range(height):
        h_block = np.hsplit(h_data[h], width)  # 图像水平分割
        for w in range(width):
            block = h_block[w]
            cover_image_blocks[h, w, :, :, :] = block

            key1 = watermark_binary1[h, w]
            key2 = watermark_binary2[h, w]
            key3 = watermark_binary3[h, w]
            for i in range(3):
                b = block[:, :, i]
                coeffs = pywt.dwt2(b, 'haar')
                LL, (LH, HL, HH) = coeffs
                LL_dct = cv2.dct(LL)
                U, S, V = np.linalg.svd(LL_dct, full_matrices=True)
                S_dft = fft(S)

                if i == 0:
                    if (w + h + key1) % 2 == 0:
                        S_dft += alpha1
                    else:
                        S_dft -= alpha1
                elif i == 1:
                    if (w + h + key2) % 2 == 0:
                        S_dft += alpha2
                    else:
                        S_dft -= alpha2
                elif i == 2:
                    if (w + h + key3) % 2 == 0:
                        S_dft += alpha3
                    else:
                        S_dft -= alpha3

                S = ifft(S_dft).real
                LL_dct = np.dot(U, np.dot(np.diag(S), V))
                LL = cv2.idct(LL_dct)
                coeffs = LL, (LH, HL, HH)
                b = pywt.idwt2(coeffs, 'haar')
                embed_image_blocks[h, w, :, :, i] = b
                embed_image_yuv[h * blocksize:(h + 1) * blocksize, w * blocksize:(w + 1) * blocksize, i] = b
    embed_image_yuv = np.clip(embed_image_yuv, 0, 255).astype(np.uint8)

    # YUV转化回BGR
    embed_image = cv2.cvtColor(embed_image_yuv, cv2.COLOR_YUV2BGR)
    return embed_image, watermark


if __name__ == '__main__':
    cover = cv2.imread('./final image/original nwpu.jpg')
    watermark1 = cv2.imread('./final image/watermark1.jpg')
    cv2.imshow('', watermark1)
    cv2.waitKey(0)
    watermark2 = cv2.imread('./final image/watermark2.jpg')
    cv2.imshow('', watermark2)
    cv2.waitKey(0)
    watermark3 = cv2.imread('./final image/watermark3.jpg')
    cv2.imshow('', watermark3)
    cv2.waitKey(0)
    embed_image, watermark = embedding(cover, watermark1, 9.28, watermark2, 4.26, watermark3, 7.4)
    cv2.imwrite('embed nwpu.jpg', embed_image)
    cv2.imwrite('watermark.jpg', watermark)
    cv2.imshow('Embed Image', embed_image)

    cv2.waitKey(0)
    cv2.imshow('Watermark', watermark)
    cv2.waitKey(0)
