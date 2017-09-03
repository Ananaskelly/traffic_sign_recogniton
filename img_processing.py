import cv2


def load_and_process(path, width, height, x, y):
    img = cv2.imread(path)
    img_roi = img[x:x+width, y:y+height]
    img_roi = cv2.resize(img_roi, (32, 32))

    # cv2.imshow('img', img_roi)
    # cv2.waitKey()
    return cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
