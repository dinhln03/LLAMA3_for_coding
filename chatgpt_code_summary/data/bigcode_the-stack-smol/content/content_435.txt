"""Python 3.9.5"""
import cv2

import HandTrackingModule as htm


def thumbIncrementCheck(lmList: list[list[int]]) -> int:
    """Checks whether your thumb is up or not.
    No matter what hand you use.
    returns 1 if thumb is up else 0"""
    count = 0
    t_x = lmList[4][1]
    p_x = lmList[17][1]
    if t_x > p_x:  # If true: RIGHT hand
        if lmList[4][1] >= lmList[2][1]:
            count += 1
    else:  # ELse: LEFT hand
        if lmList[4][1] <= lmList[2][1]:
            count += 1
    return count


def textOutput(count, cc) -> str:
    """Returns an appropriate text output depending on
    `count` and `cc`."""
    text = "NOTHING"
    if (count, cc) == (2, 2):
        text = "SCISSOR"
    elif count == 0:
        text = "ROCK"
    elif count == 5:
        text = "PAPER"
    else:
        pass
    return text


def main():
    # cap = cv2.VideoCapture(0)   # opens the camera
    detector = htm.HandDetector()

    while True:
        success, img = cv2.imread("/home/laughinglouds/Pictures/Webcam/2021-04-13-133250.jpg")
        img = detector.findHands(img)
        lmlist = detector.findPosition(img, draw=True)

        # If a hand is not detected value will be 0
        # else non-zero (21)
        hand_exists = len(lmlist)

        tipIDs = [4, 8, 12, 16, 20] # Represents fingertips
        dipIDs = [2, 7, 11, 15, 19] # Represents landmarks below the tips
        count = 0   # keeps count of how many fingers are up
        cc = 0  # for later checking if `Scissor` or not
        if hand_exists:
            # Looping for the five fingers
            for i in range(0, 5):
                if i == 0:
                    count += thumbIncrementCheck(lmlist)
                else:
                    # 8: Index finger
                    # 12: Middle finger
                    if (lmlist[tipIDs[i]][2] < lmlist[dipIDs[i]][2]) and (
                        tipIDs[i] in (8, 12) # if either index or middle
                    ):
                        count += 1
                        cc += 1
                    elif lmlist[tipIDs[i]][2] < lmlist[dipIDs[i]][2]:
                        count += 1
                    # print(cc)
        else:
            count = -1

        txt = textOutput(count, cc)

        # (10, 140) is coordinate of txt on the screen
        cv2.putText(img, str(txt), (10, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        
        cv2.imshow("Image", img)
        # close key isn't working for me
        # os: linux mint 20.1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
