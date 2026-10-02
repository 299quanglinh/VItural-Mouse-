import cv2
import numpy as np 
import HandTrackingModule as htm 
import time 
import autopy 

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 10
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionCon=0.7)
wScr, hScr = autopy.screen.size()
print(f'Screen size: {wScr}, {hScr}')
while True:
    # 1. Tìm các điểm mốc bàn tay (Hand Landmarks)
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Lấy tọa độ đầu ngón trỏ và ngón cái
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # Đầu ngón trỏ (Landmark 8)
        x2, y2 = lmList[4][1:]   # Đầu ngón cái (Landmark 4)
        #print(x1, y1, x2, y2)

        # 3. Kiểm tra xem những ngón tay nào đang giơ lên
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img,(frameR, frameR),(wCam - frameR, hCam - frameR),(255,0,255),2)
        # 4. Chỉ có ngón trỏ giơ lên: Chế độ Di chuyển chuột (Moving Mode)
        if fingers[1] == 1 and fingers[0] == 0:
            # 5. Chuyển đổi tọa độ từ khung camera sang độ phân giải màn hình
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # Giới hạn tọa độ không vượt quá mép màn hình
            x3 = np.clip(x3, 0, wScr - 1)
            y3 = np.clip(y3, 0, hScr - 1)
            # 6. Làm mượt giá trị tọa độ (giúp chuột không bị rung giật)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # 7. Di chuyển con trỏ chuột
            autopy.mouse.move(clocX, clocY)
            plocX, plocY = clocX, clocY
            cv2.circle(img, (x1, y1), 7, (255, 255, 0), cv2.FILLED)
        # 8. Cả ngón trỏ và ngón cái đều mở: Chế độ Nhấp chuột (Clicking Mode)
        if fingers[1] == 1 and fingers[0] == 1:
            # 9. Tính khoảng cách giữa đầu ngón trỏ và ngón cái
            length, img, lineInfo = detector.findDistance(4, 8, img)
            print(length)
            # 10. Thực hiện nhấp chuột (click) nếu khoảng cách giữa 2 ngón tay đủ gần
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 7, (0, 0, 255), cv2.FILLED)
                autopy.mouse.click()
    # 11. Tính toán và hiển thị tốc độ khung hình (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    # 12. Hiển thị hình ảnh ra màn hình
    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
