import cv2
import numpy as np

def main():
    img = cv2.imread('test.jpg') 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = img_gray.copy()

    param_sets = [
        (9, 3, 50),
        (11, 5, 75),
        (7, 2, 25)
    ]

    for (n, sigma_s, sigma_r) in param_sets:
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                filtered_value = bilateralFilter(img_gray, n, i, j, sigma_s, sigma_r)
                print(f'i: {i}, j: {j}, filtered_value: {filtered_value}')
                res[i, j] = filtered_value
        cv2.imwrite(f'n:{n},sigma_s:{sigma_s},sigma_r:{sigma_r}.jpg', res) 

def gaussian(x, sigma):
    return np.exp(-(x**2)/(2*sigma**2))

def Wp(img_gray, n, i, j, G_s, G_r):
    half_n = n // 2
    if n % 2 == 0:
        half_n -= 1  # Adjust for even kernel size
    sum_weights = 0

    for i_n in range(-half_n, half_n + 1):
        ni = i + i_n
        if 0 <= ni < img_gray.shape[0]:
            for j_n in range(-half_n, half_n + 1):
                nj = j + j_n
                if 0 <= nj < img_gray.shape[1]:
                    spatial_distance = np.sqrt(i_n ** 2 + j_n ** 2)
                    photometric_distance = int(img_gray[ni, nj]) - int(img_gray[i, j])

                    g1 = gaussian(spatial_distance, G_s)
                    g2 = gaussian(photometric_distance, G_r)

                    sum_weights += g1 * g2

    return sum_weights

def bilateralFilter(img_gray, n, i, j, G_s, G_r):
    wp = Wp(img_gray, n, i, j, G_s, G_r)
    if wp == 0:
        return img_gray[i, j]

    half_n = n // 2
    if n % 2 == 0:
        half_n -= 1  # Adjust for even kernel size
    sum_filtered = 0

    for i_s in range(-half_n, half_n + 1):
        si = i + i_s
        if 0 <= si < img_gray.shape[0]:
            for j_s in range(-half_n, half_n + 1):
                sj = j + j_s
                if 0 <= sj < img_gray.shape[1]:
                    spatial_distance = np.sqrt(i_s ** 2 + j_s ** 2)
                    photometric_distance = int(img_gray[si, sj]) - int(img_gray[i, j])

                    g1 = gaussian(spatial_distance, G_s)
                    g2 = gaussian(photometric_distance, G_r)

                    sum_filtered += img_gray[si, sj] * g1 * g2 

    return sum_filtered / wp

if __name__ == "__main__":
    main()
