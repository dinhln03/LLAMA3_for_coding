from colored import fg, stylize, attr
import requests as rq
from yaspin import yaspin

version = "0.4beta"

greeting = stylize("""
    ╭────────────────────────────────────────────────────────────────╮
    │ Добро пожаловать в                                             │
    │             _____ _  _             ____ _     ___              │
    │            | ____| |(_)_   _ _ __ / ___| |   |_ _|             │
    │            |  _| | || | | | | '__| |   | |    | |              │
    │            | |___| || | |_| | |  | |___| |___ | |              │
    │            |_____|_|/ |\__,_|_|   \____|_____|___|             │
    │                   |__/                                         │
    │ вер. 0.6.1beta                                                 │
    ╰────────────────────────────────────────────────────────────────╯

""", fg("magenta"), attr("bold"))

API_URL = "https://markbook.eljur.ru/apiv3/"
DEVKEY = "9235e26e80ac2c509c48fe62db23642c"
VENDOR = "markbook"

lessons = []

time_style = fg("green") + attr("bold")
room_style = fg("yellow") + attr("bold")
day_of_week_style = fg("orange_1") + attr("bold")

non_academ_style = fg("cyan")

separator_style = fg("medium_purple_1") + attr("bold")
separator = stylize("::", separator_style)

# yakuri354 - Для обозначения времени окон
# butukay - Я бы назвал это костылём                            } < Немогу удалить
# yakuri354 ~> ну я согласен, но а как ещё окна отображать

lessons_time = {
    "1": "08:30:00_09:10:00",
    "2": "09:30:00_10:10:00",
    "3": "10:20:00_11:00:00",
    "4": "11:10:00_11:50:00",
    "5": "12:00:00_12:40:00",
    "6": "13:30:00_14:10:00",
    "7": "14:20:00_15:00:00",
    "8": "15:10:00_15:50:00",
    "9": "16:20:00_17:00:00",
    "10": "17:10:00_17:50:00",
    "11": "18:00:00_18:40:00"
}


# Объект ученика
class Student:
    def __init__(self, token=None, login=None):
        self.token = token
        self.login = login

        rules_params = {
            "DEVKEY": DEVKEY,
            "vendor": VENDOR,
            "out_format": "json",
            "auth_token": self.token,
        }

        user_info = rq.get(API_URL + "getrules", params=rules_params).json()["response"]

        if user_info["error"] is not None or "":
            print("Ошибка при получении информации об ученике: " + user_info["error"])

            raise LookupError(user_info["error"])

        self.student_id = user_info["result"]["name"]

        self.name = user_info["result"]["relations"]["students"][self.student_id]["title"]
        self.grade = user_info["result"]["relations"]["students"][self.student_id]["class"]

        self.city = user_info["result"]["city"]
        self.email = user_info["result"]["email"]
        self.fullname = user_info["result"]["title"]
        self.gender = user_info["result"]["gender"]
        self.school = user_info["result"]["relations"]["schools"][0]["title"]

    def __str__(self):

        text = ""
        text += "\nИмя: " + self.name
        text += "\nКласс: " + str(self.grade)
        text += "\nГород: " + self.city
        text += "\nШкола: " + self.school
        text += "\nПол: " + "Мужской" if self.gender == "male" else "Женский"
        text += "\nЛогин: " + self.login
        text += "\nЭл. Почта: " + self.email

        return text

    def get_schedule(self, date=None, silent=False):

        load_spinner = None

        if not silent:
            load_spinner = yaspin(text="Загрузка...")
            load_spinner.text = "[Получение дневника из журнала...]"

        if date is None:
            date = "20191118-20191124"

        diary = rq.get(
            API_URL + "getschedule",
            params={
                "devkey": DEVKEY,
                "vendor": VENDOR,
                "out_format": "json",
                "student": self.student_id,
                "auth_token": self.token,
                "days": date,
                "rings": "true"
            }
        ).json()['response']

        if diary["error"] is not None:
            if not silent:
                load_spinner.text = ""
                load_spinner.fail(stylize("Ошибка получения расписания: " + diary["error"], fg("red")))

                raise LookupError(diary["error"])

        schedule = diary['result']['students'][str(self.student_id)]

        if not silent:
            load_spinner.text = ""
            load_spinner.ok(stylize("[Расписание успешно получено!] ", fg("green")))

        return schedule

    # Получение информации об ученике через запрос getrules
    def info(self, extended=False):

        if not extended:
            return self.student_id, self.name, self.grade
        else:
            return {
                "student_id": self.student_id,
                "fullname": self.name,
                "grade": self.grade,
                "city": self.city,
                "email": self.email,
                "gender": self.gender,
                "school": self.school
            }
