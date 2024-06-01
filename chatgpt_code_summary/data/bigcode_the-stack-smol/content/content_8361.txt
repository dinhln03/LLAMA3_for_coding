import time
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from flask.views import MethodView

from config.setting import BOT_TOKEN
from models import User

hitcon_zeroday_base_url = "https://zeroday.hitcon.org"
hitcon_zeroday_all_url = "https://zeroday.hitcon.org/vulnerability/all"
hitcon_zeroday_disclosed_url = "https://zeroday.hitcon.org/vulnerability/disclosed/"


def get_last_page_num(hitcon_zeroday_url):
    r = requests.get(hitcon_zeroday_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    try:
        return int(soup.find("span", {"class": "last-page"}).text)
    except Exception:
        return 0


def get_report_info(report_url):
    r = requests.get(report_url)
    if r.status_code != 200:
        return {}
    
    soup = BeautifulSoup(r.text, 'html.parser')
    last_update_str = soup.find("div", {"class": "status-descr"}).text
    last_update_date = datetime.strptime(last_update_str, r"Last Update : %Y/%m/%d").date()

    # Get utf+8 datetime
    now_utc_plus_8 = datetime.utcnow() + timedelta(hours=8)
    # Get only yesterday's data
    now_date = datetime.strptime((now_utc_plus_8 - timedelta(days=1)).strftime("%Y/%m/%d"), "%Y/%m/%d").date()
    if now_date != last_update_date:
        return {}

    data = {
        "status": soup.find("div", {"class": "status-label"}).text
    }
    for li in soup.find("div", {"class": "info"}).findAll("li"):
        if "風險" in li.text:
            data["risk"] = li.text.split("：")[-1]
        elif "類型" in li.text:
            data["type"] = li.text.split("：")[-1]
    return data


def search_page(hitcon_zeroday_base_url, hitcon_zeroday_category_url):
    last_page_num = get_last_page_num(hitcon_zeroday_category_url)
    msg_list = []
    msg_list_len = len(msg_list)
    for page_num in range(1, last_page_num+1):
        page_url = urljoin(hitcon_zeroday_category_url, f"page/{page_num}")
        r = requests.get(page_url)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, 'html.parser')

        # parse all blocks
        for li in soup.findAll("li", {"class": "strip"}):
            a = li.find("h4").find("a")
            report_url = urljoin(hitcon_zeroday_base_url, a["href"])
            title = a.text
            _data = get_report_info(report_url)
            if _data:
                msg_list.append(f"[[{_data['status']} - {_data['risk']}]] {_data['type']}")
                msg_list.append(f"[{title}]({report_url})")
        
        # break if not append new data
        if len(msg_list) == msg_list_len:
            break
        msg_list_len = len(msg_list)
    return msg_list

def send_message(chat_id, msg):
    api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&disable_web_page_preview=1&text={msg}"
    requests.get(api_url)

class App(MethodView):
    def get(self):
        msg_list = search_page(hitcon_zeroday_base_url, hitcon_zeroday_all_url)
        report_msg = "%0A".join(msg_list)
        for user in User.get_all():
            print(user, report_msg)
            send_message(user["chat_id"], report_msg)
        return "OK"
