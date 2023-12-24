import cv2 as cv
import numpy as np
from pywt import dwt2
from skimage import feature as fd
import pickle
from time import perf_counter

def load_model(model_name):
    with open(model_name, 'rb') as open_file:
        model = pickle.load(open_file)
    return model

def get_gray(image):
    b, g, r = cv.split(image)
    r = np.where(r == g, r, 255)
    r = np.where(r == b, r, 255)
    res = cv.merge([r, r, r])
    return res

def overlap3(image1, image2):
    x, y, z = cv.split(image1)
    xp, yp, zp = cv.split(image2)
    x = cv.bitwise_and(x, xp)
    y = cv.bitwise_and(y, yp)
    z = cv.bitwise_and(z, zp)
    image = cv.merge([x, y, z])
    return image

def overlap1(image, mask):
    b, g, r = cv.split(image)
    b = cv.bitwise_and(b, mask)
    g = cv.bitwise_and(g, mask)
    r = cv.bitwise_and(r, mask)
    image = cv.merge([b, g, r])
    return image

def count_white(image):
    counter = np.sum(image == 255)
    return counter

def update_list(record, plist):
    for i in plist:
        if i[0] == record[0] and i[1] == record[1] and i[2] == record[2] and i[3] == record[3]:
            if i[4] >= record[4]:
                i[5] += 1
            else:
                i[5] -= 1
                if i[5] < 0:
                    plist.remove(i)
            plist.append(record)
            return
        if abs(i[0] - record[0]) < 20 and abs(i[1] - record[1]) < 20:
            i[2] += record[2] + abs(i[2] - record[2])
            i[3] += record[3] + abs(i[3] - record[3])
            i[0] = min(i[0], record[0])
            i[1] = min(i[1], record[1])
            plist.append(record)
            return
    plist.append(record)

def normalize_0_1(oneDarray):
    return oneDarray / (max(oneDarray) - min(oneDarray))

def normalization(array3D):
    R, G, B = cv.split(array3D)
    I = R.astype(float) + G.astype(float) + B.astype(float)
    I = np.where(I == 0, I + 1, I)
    r, g, b = R / I, G / I, B / I
    rgb = cv.merge([b, g, r])
    return rgb

def feature_extractor(image):
    features = []

    HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    LAB = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    R, G, B = cv.split(image)
    H, S, V = cv.split(HSV)
    L, A, B1 = cv.split(LAB)

    histograms = [np.histogram(channel, bins=8)[0] for channel in (R, G, B, H, S, A, B1)]
    for histogram in histograms:
        features = np.append(features, normalize_0_1(histogram))

    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, (cH, cV, cD) = dwt2(im, 'db1')
    energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / im.size
    histogramE = np.histogram(energy, bins=8)[0]
    features = np.append(features, normalize_0_1(histogramE))

    fft = np.fft.fft2(image)
    fshift = np.fft.fftshift(fft)
    abs1 = abs(fshift)
    log1 = np.log(abs1 + 0.00000001)
    flog = abs(normalization(log1))
    histogramFFT = np.histogram(flog * 255, bins=8)[0]
    features = np.append(features, normalize_0_1(histogramFFT))

    lbps = [fd.local_binary_pattern(channel, 24, 8, method="uniform") for channel in (R, G, B)]
    for lbp in lbps:
        histogramLBP = np.histogram(lbp.ravel(), bins=8)[0]
        features = np.append(features, normalize_0_1(histogramLBP))

    mean = np.mean(image)
    var = np.var(image)
    haziness = calculate_haziness(image)
    features = np.append(features, mean, var, haziness)

    cn1 = np.amax(V) - np.amin(V)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cn2 = np.amax(gray) - np.amin(gray)
    features = np.append(features, cn1, cn2)

    v_min, v_max, v_mean, v_means, v_standard = cv.meanStdDev(V)
    features = np.append(features, v_min, v_max, v_mean, v_means, v_standard)

    return features

def calculate_haziness(image):
    mu, v, sicma, landa = 5.1, 2.9, 0.2461, 1 / 3
    di = np.min(image, axis=2)
    bi = np.max(image, axis=2)
    d = np.mean(np.mean(di))
    b = np.mean(np.mean(bi))
    c = b - d
    A = landa * np.max(np.max(bi)) + (1 - landa) * b + 0.00001
    x1 = (A - d) / A
    x2 = c / A
    haziness = np.exp(-0.5 * (mu * x1 + v * x2) + sicma)
    return haziness

def process_frame(frame, bc, kernel, model, plist):
    frame = cv.resize(frame, (640, 480))
    fg_mask = bc.apply(frame)
    fg_mask = cv.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv.dilate(fg_mask, kernel, iterations=1)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab)

    h, s, b = cv.split(hsv)
    l, a, b1 = cv.split(lab)

    h_mask = cv.inRange(h, 90, 120)
    s_mask = cv.inRange(s, 10, 50)
    b_mask = cv.inRange(b, 110, 129)
    mask = cv.bitwise_and(cv.bitwise_and(h_mask, s_mask), b_mask)

    injected = cv.bitwise_and(mask, fg_mask)

    _, image = cv.threshold(injected, 127, 255, cv.THRESH_BINARY)

    cnts, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv.contourArea(contour) < 400:
            continue
        x, y, width, height = cv.boundingRect(contour)
        white_pix = count_white(frame[y:y + height, x:x + width])
        record = [x, y, width, height, white_pix, 1]
        update_list(record, plist)

        region = frame[y:y + height, x:x + width]
        flist = []
        for j in range(0, region.shape[0], 25):
            for i in range(0, region.shape[1], 25):
                part = region[j:j + 25, i:i + 25]
                fe_list = feature_extractor(part)
                flist.append(fe_list)

        flist = np.array(flist)
        if flist is not None:
            predict = model.predict(flist)
            if any(predict):
                cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 1)
                cv.putText(frame, "smoke", (x, y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

def main():
    model_name = "GNB_Model-newfeatures.sav"
    model = load_model(model_name)

    input_name = "data\\smoke\\pos\\movie2.avi"
    output_name = "fireANDsmoke-2-test-1.avi"

    cap = cv.VideoCapture(input_name)
    bc = cv.createBackgroundSubtractorKNN()
    kernel = np.ones((3, 3), np.uint8)
    plist = []

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    writer = cv.VideoWriter(output_name, fourcc, 8, (640, 480), True)

    fpc = 0
    tic = perf_counter()

    while True:
        _, frame = cap.read()
        if frame is None:
            break
        processed_frame = process_frame(frame, bc, kernel, model, plist)
        writer.write(processed_frame)
        fpc += 1

        cv.imshow("Output", processed_frame)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    toc = perf_counter()
    fps = fpc / (toc - tic)
    print(f"Frames per second: {fps:.2f}")

    writer.release()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
