import numpy as np
import cv2 as cv

img_l = cv.imread('left.png', 0)
img_r = cv.imread('right.png', 0)

def stereoPM():
    # stereo = cv.StereoBM_create(
    #     numDisparities=48,
    #     blockSize=11)
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=5,
        uniquenessRatio=1,
        speckleRange=3,
        speckleWindowSize=3,
        disp12MaxDiff=200,
        P1=600,
        P2=2400)
    disp = stereo.compute(img_l, img_r).astype(np.float32)
    res = disp / np.max(disp)
    return res

def stereoLBP(img_l, img_r, k=50, s=0.05, eta=1, iterations=20):
    H, W, C = img_l.shape
    D = np.zeros((H, W, k))
    for i in range(k):
        tmp = np.zeros((H, W, C))
        tmp[:, :W - i, :] = img_l[:, i:, :]
        D[:, :, i] = np.sum((img_r - tmp) ** 2, -1) ** 0.5
    D = D / np.max(D)

    AFFINE_DIR = {'up': np.array([[1, 0, 0], [0, 1, -1]], dtype=np.float32),
                  'down': np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32),
                  'left': np.array([[1, 0, -1], [0, 1, 0]], dtype=np.float32),
                  'right': np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)}
    m = {'up':      np.zeros((H, W, k)),
         'down':    np.zeros((H, W, k)),
         'left':    np.zeros((H, W, k)),
         'right':   np.zeros((H, W, k))}
    h = {'up':      np.zeros((H, W, k)),
         'down':    np.zeros((H, W, k)),
         'left':    np.zeros((H, W, k)),
         'right':   np.zeros((H, W, k))}
    for _ in range(iterations):
        h_tot = D + m['up'] + m['down'] + m['left'] + m['right']
        h['up'] = cv.warpAffine(h_tot-m['down'], AFFINE_DIR['down'], dsize=(W, H))
        h['down'] = cv.warpAffine(h_tot-m['up'], AFFINE_DIR['up'], dsize=(W, H))
        h['left'] = cv.warpAffine(h_tot-m['right'], AFFINE_DIR['right'], dsize=(W, H))
        h['right'] = cv.warpAffine(h_tot-m['left'], AFFINE_DIR['left'], dsize=(W, H))
        for x in {'up', 'down', 'left', 'right'}:
            m[x] = h[x]
            for i in range(1, k):
                m[x][:, :, i] = np.minimum(m[x][:, :, i], m[x][:, :, i-1] +  s)
            for i in reversed(range(0, k-1)):
                m[x][:, :, i] = np.minimum(m[x][:, :, i], m[x][:, :, i+1] + s)

        for x in {'up', 'down', 'left', 'right'}:
            tmp = h[x].min(axis=-1, keepdims=True) + eta
            m[x] = np.minimum(m[x], tmp)
    B = np.copy(D)
    for x in {'up', 'down', 'left', 'right'}:
        B = B + m[x]
    tmp =  np.argmin(B, -1)
    res = tmp / np.max(tmp)
    return res

if __name__ == '__main__':
    img1 = cv.imread('left.png')
    img2 = cv.imread('right.png')
    disp = stereoPM(img1, img2)
    # disp = stereoLBP(img1, img2)
    cv.imshow('Depth-Image', disp)
    cv.waitKey(0)


