from .filters import SOBEL_X, SOBEL_Y

def convolve2d(img, kernel):
    h, w = len(img), len(img[0])
    kh, kw = len(kernel), len(kernel[0])

    out = [[0]*(w-kw+1) for _ in range(h-kh+1)]

    for i in range(len(out)):
        for j in range(len(out[0])):
            s = 0
            for ki in range(kh):
                for kj in range(kw):
                    s += img[i+ki][j+kj]*kernel[ki][kj]
            out[i][j] = s

    return out


def edge(img):
    gx = convolve2d(img, SOBEL_X)
    gy = convolve2d(img, SOBEL_Y)

    return [
        [
            (gx[i][j]**2 + gy[i][j]**2)**0.5
            for j in range(len(gx[0]))
        ]
        for i in range(len(gx))
    ]