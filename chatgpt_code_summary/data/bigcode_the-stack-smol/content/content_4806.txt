import pandas as pd
import datetime as dt
import smtplib as st
import random as rd

FROM = "pythonsender633@gmail.com"
PASSWORD = "1234abc()"
SUBJECT = "Happy birthday!"

LETTERS = [1, 2, 3]
PLACEHOLDER = "[NAME]"

PATH = "birthdays.csv"
C_NAME = "name"
C_EMAIL = "email"
C_YEAR = "year"
C_MONTH = "month"
C_DAY = "day"

data = pd.read_csv(PATH)
current = dt.datetime.now()

for row in data.iterrows():
    row = row[1]
    birthday = dt.datetime(int(row[C_YEAR]), int(row[C_MONTH]), int(row[C_DAY]))

    if current.month == birthday.month and current.day == birthday.day:
        number = rd.choice(LETTERS)

        with open(f"letter_templates/letter_{number}.txt") as handle:
            letter = handle.read()
            letter = letter.replace(PLACEHOLDER, row[C_NAME])

        with st.SMTP("smtp.gmail.com") as connection:
            message = f"Subject:{SUBJECT}\n\n{letter}"

            connection.starttls()
            connection.login(user=FROM, password=PASSWORD)
            connection.sendmail(
                from_addr=FROM,
                to_addrs=row[C_EMAIL],
                msg=message
            )
