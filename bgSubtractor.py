import cv2

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
    ret, frame = captura.read()
    if ret == False:
        break

    MOGmask = mog.apply(frame)
    MOG2mask = mog2.apply(frame)
    gmgmask = gmg.apply(frame)

    cv2.imshow('Video de prueba', frame)
    cv2.imshow('MOG',MOGmask)
    cv2.imshow('MOG2',MOG2mask)
    cv2.imshow('GMG',gmgmask)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break 
captura.release()
cv2.destroyAllWindows()
