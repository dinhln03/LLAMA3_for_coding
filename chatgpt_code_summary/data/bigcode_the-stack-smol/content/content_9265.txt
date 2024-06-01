# 發送信件
import os
import sys
import smtplib
import getpass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

COMMASPACE = ', '

def main():
    sender = input("Please enter your email here:")
    gmail_password = getpass.getpass("Please enter your password here:")
    recipients = input("Who would you want to send to ?:")
    recipients_list = []
    recipients_list.append(recipients)
    # 建立郵件主題
    outer = MIMEMultipart()
    outer['Subject'] = '日本東京都內最新疫情'
    outer['To'] = COMMASPACE.join(recipients_list)
    outer['From'] = sender
    outer.preamble = 'You will not see this in a MIME-aware mail reader.\n'

    # 檔案位置
    attachments = ['/Users/bomin/Documents/tokyocovid19crawler/covid19.csv']

    # 加入檔案到MAIL底下
    for file in attachments:
        try:
            with open(file, 'rb') as fp:
                print ('can read file')
                msg = MIMEBase('application', "octet-stream")
                msg.set_payload(fp.read())
            encoders.encode_base64(msg)
            msg.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file))
            outer.attach(msg)
        except:
            print("Unable to open one of the attachments. Error: ", sys.exc_info()[0])

    composed = outer.as_string()

    # 寄送EMAIL
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(sender, gmail_password)
            s.sendmail(sender, recipients_list, composed)
            s.close()
        print("Email sent!")
    except:
        print("Unable to send the email. Error: ", sys.exc_info()[0])


main()