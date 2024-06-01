from tkinter import *
from tkinter import ttk
import time
import time

window = Tk()

mygreen = "lightblue"
myred = "blue"

style = ttk.Style()

style.theme_create( "dedoff", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {
            "configure": {"padding": [5, 1], "background": mygreen },
            "map":       {"background": [("selected", myred)],
                          "expand": [("selected", [1, 1, 1, 0])] } } } )

style.theme_use("dedoff")


window.title("Электронный учебник tkinter")
window.geometry('1920x1080')
tab_control = ttk.Notebook(window)

#панели

tab1 = ttk.Frame(tab_control, width=1920, height=1080)
tab2 = ttk.Frame(tab_control, width=1920, height=1080)
tab3 = ttk.Frame(tab_control, width=1080, height=600)
tab4 = ttk.Frame(tab_control, width=1080, height=600)
tab5 = ttk.Frame(tab_control, width=1080, height=600)
tab6 = ttk.Frame(tab_control, width=1080, height=600)
tab7 = ttk.Frame(tab_control, width=1080, height=600)
tab8 = ttk.Frame(tab_control, width=1080, height=600)
tab9 = ttk.Frame(tab_control, width=1080, height=600)
tab10 = ttk.Frame(tab_control, width=1080, height=600)

tab_control.add(tab1, text='Начало')

background_image = PhotoImage(file='background.ppm')
background_label = Label(tab1, image=background_image)
background_label.place(relwidth=1, relheight=1)

lower_frame = Frame(tab1, bg="lightblue", bd=10)
lower_frame.place(relx=0.5, rely=0.10, relwidth=0.75, relheight=0.75, anchor='n')

labeltext1 = Label(lower_frame, text="Tkinter – это кроссплатформенная библиотека для разработки графического интерфейса на  "
                              "языке Python\n (начиная с Python 3.0 переименована в tkinter). Tkinter расшифровывается "
                              "как Tk interface \nНачиная с версии python-3.0 библиотека переименована в соответствии с "
                              "PEP 8 в tkinter (с маленькой буквы). \nИмпортируется она как и любая другая библиотека "
                              "абсолютно весь код в этом учебнике написан для python версии 3.x \nПодключить модуль "
                              "можно с помощью инструкции import. После ключевого слова import указывается название "
                              "модуля.\n Одной инструкцией можно подключить несколько модулей. Для подключения всех \n"
                              "функций модуля используем:\n"
                              "import tkinter \n"
                              "или \n"
                              "from tkinter import * \n"
                              "Чтобы убедиться, что Tkinter установлен и работает, воспользуемся стандартной "
                              "функцией Tkinter: test():"
                                     "\n"
                                     "import tkinter \n"
                                     "tkinter._test() \n"
                   ,
                   font=("Times New Roman", 13), bg="white")
labeltext1.place(relwidth=1, relheight=0.6)
photo = PhotoImage(file='edu54img.pgm')
labelimage = Label(lower_frame,bg='white', image=photo)
labelimage.place(relx=0.5, rely=0.6, relwidth=1, relheight=0.4, anchor='n')

#ОГО ВТОРООООООООООЙ ТААААААААААААААААААБ

tab_control.add(tab2, text='Canvas')

background_image2 = PhotoImage(file='background.ppm')
background_label1 = Label(tab2, image=background_image2)
background_label1.place(relwidth=1, relheight=1)

lower_frame1 = Frame(tab2, bg="lightblue", bd=10)
lower_frame1.place(relx=0.5, rely=0.02, relwidth=0.75, relheight=0.95, anchor='n')

labeltext2 = Label(lower_frame1, text=u"Привет, это второй раздел учебника.\n В tkinter от класса Canvas создаются объекты-холсты, на которых можно рисовать,\n"
                              "размещая различные фигуры и объекты. Делается это с помощью вызовов соответствующих \n"
                              "методов. При создании экземпляра Canvas необходимо указать его ширину и высоту. При \n"
                              "размещении геометрических примитивов и других объектов указываются их координаты на \n "
                              "холсте. Точкой отсчета является верхний левый угол.", font=("Times New Roman", 12), bg="white")
labeltext2.place(relwidth=1, relheight=0.3)
photo2 = PhotoImage(file='edu54img2.pgm')
labelimage1 = Label(lower_frame1, bg='white', image=photo2)
labelimage1.place(relx=0.5, rely=0.30, relwidth=1, relheight=0.49, anchor='n')
labeltext2 = Label(lower_frame1, text="В программе ниже создается холст.\n"
                                      "from tkinter import *\n"
                                      "window = Tk()\n"
                                      "c = Canvas(root, width=200, height=200, bg='white')\n"
                                      "c.pack()\n"
                                      "window.mainloop()\n"
                                      "в следующей главе мы разберем как рисовать на этом холсте", font=("Times New Roman", 12), bg="white")
labeltext2.place(relx=0.5, rely=0.75, relwidth=1, relheight=0.3, anchor='n')

tab_control.add(tab3, text='Примитивы')

background_image3 = PhotoImage(file='background.ppm')
background_label2 = Label(tab3, image=background_image3)
background_label2.place(relwidth=1, relheight=1)

lower_frame2 = Frame(tab3, bg="lightblue", bd=10)
lower_frame2.place(relx=0.5, rely=0.02, relwidth=0.8, relheight=0.95, anchor='n')
labeltext3 = Label(lower_frame2, text="В tkinter уже есть графические примитивы, для рисования, их нужно всего лишь правильно "
                              "указать.\n В программе ниже создается холст. На нем с помощью метода create_line() "
                              "рисуются отрезки. \n Сначала указываются координаты начала (x1, y1), затем – конца (x2, "
                              "y2) В программе ниже создаётся и рисуется линия на холсте.", font=("Times New Roman", 12), bg="white")
labeltext3.place(relwidth=1, relheight=0.12)
codeimg = PhotoImage(file='code.pgm')
labelimg = Label(lower_frame2, bg='white', image=codeimg)
labelimg.place(relx=0.5, rely=0.11, relwidth=1, relheight=0.5, anchor='n')
labelgotext = Label(lower_frame2, text="Собственно сами примитивы. Указываем координаты примитива всегда следующим образом – \n "
                               "верхний левый угол(x1, y1), вторые – правый нижний(x2, y2).", font=("Times New "
                                                                                                    "Roman", 11),
                    bg='white')
labelgotext.place(relx=0.5, rely=0.52, relwidth=1, relheight=0.07, anchor='n')
rectangle = PhotoImage(file='rectangle.ppm')
rectanglelabel = Label(lower_frame2, bg='white', image=rectangle)
rectanglelabel.place(relx=0.5, rely=0.60, relwidth=1, relheight=0.45, anchor='n')
labelgotext2 = Label(lower_frame2, text="Далее о других примитивах в следующей вкладке", font=("Times New "
                                                                                                    "Roman", 11),
                     bg='white')
labelgotext2.place(relx=0.5, rely=0.97, relwidth=1, relheight=0.05, anchor='n')

tab_control.add(tab4, text='Примитивы 2')

background_image4 = PhotoImage(file='background.ppm')
background_label3 = Label(tab4, image=background_image4)
background_label3.place(relwidth=1, relheight=1)

lower_frame3 = Frame(tab4, bg="lightblue", bd=10)
lower_frame3.place(relx=0.5, rely=0, relwidth=0.9, relheight=1, anchor='n')

oval = PhotoImage(file='oval_1.ppm')
ovallabel = Label(lower_frame3,bg='white', image=oval)
ovallabel.place(relx=0.5, rely=0, relwidth=1, relheight=0.55, anchor='n')

elipsoid = PhotoImage(file='ellipssmall.ppm')
elabel = Label(lower_frame3, bg='white', image=elipsoid)
elabel.place(relx=0.5, rely=0.5, relwidth=1, relheight=0.25, anchor='n')

labeltext4 = Label(lower_frame3, text="Метод create_oval(x1, y1, x2, y2) создает эллипсы. При этом задаются координаты гипотетического "
                              "прямоугольника, описывающего эллипс. \nЕсли нужно получить круг, то соответственно "
                              "описываемый прямоугольник должен быть квадратом.\n"
                                      "Методом create_polygon(x1, x2...xn, yn) рисуется произвольный многоугольник путем задания координат каждой его точки\n"
                                      "Создание прямоугольников методом create_rectangle(x1, y1, x2, y2)\n"
                                      "Опции: \nwidth=число - ширина обводки, fill='color' - цвет заливки,\n outline='color' - цвет "
                              "обводки,\n activefill определяет цвет при наведении на него курсора мыши.\n"
                              "activeoutline определяет цвет обводки при наведении курсор", font=("Times New Roman", 11),
                   bg="white")
labeltext4.place(relx=0.5, rely=0.74, relwidth=1, relheight=0.26, anchor='n')

tab_control.add(tab5, text='Примитивы 3')

background_image5 = PhotoImage(file='background.ppm')
background_label4 = Label(tab5, image=background_image5)
background_label4.place(relwidth=1, relheight=1)

lower_frame4 = Frame(tab5, bg="lightblue", bd=10)
lower_frame4.place(relx=0.5, rely=0.05, relwidth=0.75, relheight=0.9, anchor='n')

labeltext5 = Label(lower_frame4, text="Более сложные для понимания фигуры получаются при использовании метода create_arc(). В \n"
                              "зависимости от значения опции style можно получить сектор (по умолчанию), \n"
                              "сегмент (CHORD) или дугу (ARC). Также как в случае create_oval() координаты задают \n"
                              "прямоугольник, в который вписана окружность (или эллипс), из которой вырезают сектор, \n"
                              "сегмент или дугу. Опции start присваивается градус начала фигуры, extent определяет "
                              "угол поворота.",
                   font=("Times New Roman", 11), bg="white")
labeltext5.place(relwidth=1, relheight=0.2)

arc = PhotoImage(file='arc.ppm')
arclabel = Label(lower_frame4,bg='white', image=arc)
arclabel.place(relx=0.5, rely=0.15, relwidth=1, relheight=0.4, anchor='n')

arc2 = PhotoImage(file='arc2.ppm')
arclabel2 = Label(lower_frame4,bg='white', image=arc2)
arclabel2.place(relx=0.5, rely=0.55, relwidth=1, relheight=0.5, anchor='n')



tab_control.add(tab6, text='Полезное')

background_image6 = PhotoImage(file='background.ppm')
background_label6 = Label(tab6, image=background_image6)
background_label6.place(relwidth=1, relheight=1)

table = PhotoImage(file='colortable.ppm')
tablelabel = Label(tab6,bg='lightblue', image=table)
tablelabel.place(relx=0.5, rely=0, relwidth=0.82, relheight=1, anchor='n')

tab_control.add(tab7, text='Практикум')

background_image7 = PhotoImage(file='background.ppm')
background_label7 = Label(tab7, bg='white', image=background_image7)
background_label7.place(relwidth=1, relheight=1)

lower_frame7 = Frame(tab7, bg="lightblue", bd=10)
lower_frame7.place(relx=0.5, rely=0.001, relwidth=0.65, relheight=1, anchor='n')

labelTASK1 = Label(lower_frame7, text="1) Пропеллер"
                              ":Нарисуйте пропеллер, как это показано ниже\n"
                                      "'Кто мечтает быть пилотом, очень смелый видно тот. От-от-от вин-та!'", font=("Georgia", 12,), bg='white')

labelTASK1.place(relx=0.5, rely=0, relwidth=1, relheight=0.06, anchor='n')

propeller = PhotoImage(file='propellersmall.ppm')
propelabel = Label(lower_frame7, bg='white', image=propeller)
propelabel.place(relx=0.5, rely=0.06, relwidth=1, relheight=0.55, anchor='n')

labelTASK2 = Label(lower_frame7, text="2) Торт"
                              ":Нарисуйте торт для учителя информатики.\n'Треугольник' должен пропадать при наведении курсора.'\n"
                              "'Кто сьел мой двумерный массив?!'", font=("Georgia", 12, ), bg='white')
labelTASK2.place(relx=0.5, rely=0.6, relwidth=1, relheight=0.1, anchor='n')

tort = PhotoImage(file='tortsmall.ppm')
tortlabel = Label(lower_frame7, bg='white', image=tort)
tortlabel.place(relx=0.5, rely=0.69, relwidth=1, relheight=0.35, anchor='n')

tab_control.add(tab8, text='Анимации')

background_image8 = PhotoImage(file='background.ppm')
background_label8 = Label(tab8, image=background_image8)
background_label8.place(relwidth=1, relheight=1)

lower_frame8 = Frame(tab8, bg="lightblue", bd=10)
lower_frame8.place(relx=0.5, rely=0.5, relwidth=0.59, relheight=0.5, anchor='n')
labelanimation = Label(lower_frame8, text='Методы, создающие фигуры на холсте, возвращают численные идентификаторы \n'
                                          'этих объектов, которые можно присвоить переменным,\n через которые позднее '
                                          'обращаться к созданным фигурам. \n Основной шаблон для анимации с Tkinter – написать функцию, которая рисует один кадр. \n Затем используйте что-то подобное, чтобы называть его через регулярные интервалы: \n'
                                          " def animate(self): self.draw_one_frame() self.after(100, self.animate) \n"
                                          "Как только вы вызываете эту функцию один раз,\n она будет продолжать "
                                          'рисовать кадры со скоростью десять в секунду – один раз каждые 100 '
                                          "миллисекунд.\n В следующей вкладке разберём это подробно",  font=("Times New Roman", 11),
                   bg="white")
labelanimation.place(relwidth=1, relheight=1)

WIDTH = 350
HEIGHT = 300
SIZE = 50
canvas = Canvas(tab8, width=WIDTH, height=HEIGHT, bg="blue")
canvas.pack()
color = '#6098cd'

class Ball:
    def __init__(self, tag):
        self.shape = canvas.create_oval(0, 0, SIZE, SIZE, fill=color, tags=tag)
        self.speedx = 10
        self.speedy = 15
        self.active = True

    def ball_update(self):
        canvas.move(self.shape, self.speedx, self.speedy)
        pos = canvas.coords(self.shape)
        if pos[2] >= WIDTH or pos[0] <= 0:
            self.speedx *= -1
        if pos[3] >= HEIGHT or pos[1] <= 0:
            self.speedy *= -1

global switcher
switcher = True
def cycle():
    global switcher
    canvas.tag_raise("bg")
    if switcher:
        ball2.ball_update()
        ball2.ball_update()
        canvas.tag_raise("ball")
    else:
        ball.ball_update()
        ball.ball_update()
        canvas.tag_raise("ball2")
    tab8.update_idletasks()
    switcher = not switcher
    tab8.after(40, cycle)

bg = canvas.create_rectangle(0, 0, WIDTH+1, HEIGHT+1, fill="white", tags="bg")
ball = Ball("ball")
ball.ball_update()
ball2 = Ball("ball2")

tab8.after(0, cycle)

tab_control.add(tab9, text='Анимации 2')

background_image9 = PhotoImage(file='background.ppm')
background_label9 = Label(tab9, image=background_image9)
background_label9.place(relwidth=1, relheight=1)

lower_frame9 = Frame(tab9, bg="lightblue", bd=10)
lower_frame9.place(relx=0.5, rely=0.10, relwidth=0.75, relheight=0.75, anchor='n')

labelanimation2 = Label(lower_frame9, text='Рассмотрим следующий код, отвечающий за создание анимации и после этого попрактикуемся. Собственно сам код: \n', font=("Times New Roman", 11),
                   bg="white")
labelanimation2.place(relx=0.5, rely=0, relwidth=1, relheight=0.06, anchor='n')

code_image8 = PhotoImage(file='sharcode.ppm')
code_label8 = Label(lower_frame9, bg='white', image=code_image8)
code_label8.place(relx=0.5, rely=0.06, relwidth=1, relheight=0.6, anchor='n')

labelanimation3 = Label(lower_frame9, text='В данном коде создаётся шар, который двигается. Вначале происходит '
                                           'создание холста Canvas и его "упаковка"\n, а также объекта ball, '
                                           'с помощью примитива круг. После всего этого создаётся функция, которая '
                                           'анимирует данный объект, рассмотрим её очень подробно \n '
                                           'def motion (): - создание функции с названием motion \n'
                                           'c.move(ball, 1, 0) - движение объекта на c. В самом начале при создании \n '
                                           'холста мы назвали его c, следовательно при указании движения на нём мы \n'
                                           'пишем c. move - декоратор, который указывает, что делать. В нашем случае \n'
                                           'двигаться. Но чему? В скобках указываем объект движения и его координаты \n'
                                           'движения x, y. if c.coords(ball)[2] < 300, отвечает за то, чтобы шар \n'
                                           'двигался по координате X меньше 300. root.after(10, motion) - Частота обновлений окна в милисекундах. \n'
                                           'После чего с помощью motion(), запускаем нашу функцию и само окно tkinter.', font=("Times New Roman", 10),
                   bg="white")
labelanimation3.place(relx=0.5, rely=0.65, relwidth=1, relheight=0.35, anchor='n')

tab_control.add(tab10, text='Практикум 2')

background_image10 = PhotoImage(file='background.ppm')
background_label10 = Label(tab10, image=background_image10)
background_label10.place(relwidth=1, relheight=1)

# Практикум 2_поезд

c = Canvas(tab10, width=300, height=200, bg="white")
c.place(relx=0.5, rely=0.65, relwidth=0.15, relheight=0.2, anchor='n')

vagon1 = c.create_rectangle(0, 50, 60, 90, fill='blue')
line = c.create_line(60, 70, 70, 70, fill='brown', width=6)
vagon2 = c.create_rectangle(70, 50, 130, 90, fill='blue')
relsa = c.create_line(0, 90, 300, 90, fill='gray', width=3)

def motion():
    c.move(vagon1, 1, 0)
    c.move(vagon2, 1, 0)
    c.move(line, 1, 0)
    if c.coords(vagon1)[0] < 50:
        tab10.after(20, motion)


motion()

tab_control.pack(expand=10, fill='both', padx=5, pady=5)

lower_frame9 = Frame(tab10, bg="lightblue", bd=10)
lower_frame9.place(relx=0.5, rely=0.35, relwidth=0.45, relheight=0.25, anchor='n')

labelpractic2 = Label(lower_frame9, text="Анимируйте данный скетч поезда! Исходный код создания самого скетча без холста: \n vagon1 = c.create_rectangle(0, 50, 60, 90, fill='blue'\n"
                                         "line = c.create_line(60, 70, 70, 70, fill='brown', width=6) \n"
                                         "vagon2 = c.create_rectangle(70, 50, 130, 90, fill='blue') \n"
                                         "relsa = c.create_line(0, 90, 300, 90, fill='gray', width=3) \n", bg='white', font=("Times New Roman", 11))
labelpractic2.place(relwidth=1, relheight=1)

Button(window, text='© Dedov Georgiy 2019').pack(fill='x')


window.resizable(True, True)

window.mainloop()
