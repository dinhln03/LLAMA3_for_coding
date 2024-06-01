from tkinter import *
from time import *

## 전역  변수 선언 부분 ## 
fnameList = ["jeju1.gif", "jeju2.gif", "jeju3.gif", "jeju4.gif", "jeju5.gif", "jeju6.gif", "jeju7.gif", "jeju8.gif", "jeju9.gif", "jeju10.gif"]
photoList = [None] * 9
num1,num2,num3 = 0,1,2

## 함수 선언 부분 ##
def clickNext() :
    global num1,num2,num3
    num1 += 1
    num2 += 1
    num3 += 1

    if num1 > 9 :
        num1 = 0
    if num2 > 9 :
        num2 = 0
    if num3 > 9 :
        num3 = 0

    photo = PhotoImage(file="gif/" + fnameList[num1])
    photo = photo.subsample(2, 2)
    photo1 = PhotoImage(file="gif/" + fnameList[num2])
    photo1 = photo1.subsample(4, 4)
    photo2 = PhotoImage(file="gif/" + fnameList[num3])
    photo2 = photo2.subsample(4, 4)

    pLabel.configure(image = photo)
    pLabel.image=photo
    pLabel1.configure(image = photo1)
    pLabel1.image=photo1
    pLabel2.configure(image = photo2)
    pLabel2.image=photo2

    
def clickPrev() :
    global num1,num2,num3
    num1 -= 1
    num2 -= 1
    num3 -= 1

    if num1 < 0 :
        num1 = 9
    if num2 < 0 :
        num2 = 9
    if num3 < 0 :
        num3 = 9

    photo = PhotoImage(file="gif/" + fnameList[num1])
    photo = photo.subsample(2, 2)
    photo1 = PhotoImage(file="gif/" + fnameList[num2])
    photo1 = photo1.subsample(4, 4)
    photo2 = PhotoImage(file="gif/" + fnameList[num3])
    photo2 = photo2.subsample(4, 4)

    pLabel.configure(image = photo)
    pLabel.image=photo
    pLabel1.configure(image = photo1)
    pLabel1.image=photo1
    pLabel2.configure(image = photo2)
    pLabel2.image=photo2


def clickFirst():
    global num1,num2,num3
    num1,num2,num3 = 0, 9 , 1
    photo = PhotoImage(file="gif/" + fnameList[num1])
    photo = photo.subsample(2, 2)
    photo1 = PhotoImage(file="gif/" + fnameList[num2])
    photo1 = photo1.subsample(4, 4)
    photo2 = PhotoImage(file="gif/" + fnameList[num3])
    photo2 = photo2.subsample(4, 4)

    pLabel.configure(image=photo)
    pLabel.image = photo
    pLabel1.configure(image=photo1)
    pLabel1.image = photo1
    pLabel2.configure(image=photo2)
    pLabel2.image = photo2

def clickEnd() :
    global num1,num2,num3
    num1,num2,num3 = 9, 8 ,0
    photo = PhotoImage(file="gif/" + fnameList[num1])
    photo = photo.subsample(2, 2)
    photo1 = PhotoImage(file="gif/" + fnameList[num2])
    photo1 = photo1.subsample(4, 4)
    photo2 = PhotoImage(file="gif/" + fnameList[num3])
    photo2 = photo2.subsample(4, 4)

    pLabel.configure(image = photo)
    pLabel.image=photo
    pLabel1.configure(image = photo1)
    pLabel1.image=photo1
    pLabel2.configure(image = photo2)
    pLabel2.image=photo2

    
## 메인 코드 부분
window = Tk()
window.geometry("730x330")
window.title("사진 앨범 보기")
window.configure(background="white")

btnPrev = Button(window, text = "<< 이전", command = clickPrev, width = 10, background="skyblue")
btnNext = Button(window, text = "다음 >>", command = clickNext,  width = 10, background="skyblue")
btnFirst = Button(window, text = "처  음", command = clickFirst, width = 10, background="skyblue")
btnEnd = Button(window, text = "마지막", command = clickEnd, width = 10, background="skyblue")

photo = PhotoImage(file = "gif/" + fnameList[0])
photo = photo.subsample(2,2)
pLabel = Label(window, image = photo)

photo1 = PhotoImage(file = "gif/" + fnameList[9])
photo1 = photo1.subsample(4,4)
pLabel1 = Label(window, image = photo1)

photo2 = PhotoImage(file = "gif/" + fnameList[1])
photo2 = photo2.subsample(4,4)
pLabel2 = Label(window, image = photo2)

btnPrev.place(x = 280, y = 270)
btnNext.place(x = 380, y = 270)

btnFirst.place(x = 180, y = 270)
btnEnd.place(x = 480, y = 270)

pLabel1.place(x = 20, y = 50)
pLabel.place(x = 200, y = 10)
pLabel2.place(x = 545, y = 50)

window.mainloop()
