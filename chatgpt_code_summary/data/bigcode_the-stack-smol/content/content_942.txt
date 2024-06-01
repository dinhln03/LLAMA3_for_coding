from tourapi.list import TourAPI
from tourapi.config import ServiceKey, MobileOS, MobileApp, Languages
from mysql_config import MysqlHost, MysqlUser, MysqlPass, MysqlDB
import pymysql
import json

def upload_category_codes(codes, language="Kor", level=0, cat1="", cat2="", cat3=""):
  global conn, curs

  query = """
  INSERT INTO category_code (code, cat1, cat2, cat3, level, name_{0}) VALUES (%s, %s, %s, %s, %s, %s)
  ON DUPLICATE KEY UPDATE name_{0}=%s
  """.format(language.lower())

  for code in codes:
    curs.execute(query, (code["code"], cat1, cat2, cat3, level, code["name"], code["name"]))
    # print(code["name"], code["code"])
  conn.commit()

  return


conn = pymysql.connect(host = MysqlHost, user = MysqlUser, password = MysqlPass, db = MysqlDB)
curs = conn.cursor()

for lan in Languages:
  language = lan["code"]
  api = TourAPI(ServiceKey, language)

  # 대분류 카테고리
  cat1_codes = api.list_category_code()
  upload_category_codes(cat1_codes, language, 1)

  for cat1 in cat1_codes:
    cat2_codes = api.list_category_code(cat1["code"])
    upload_category_codes(cat2_codes, language, 2, cat1["code"])
    print(cat2_codes)

    for cat2 in cat2_codes:
      cat3_codes = api.list_category_code(cat1["code"], cat2["code"])
      upload_category_codes(cat3_codes, language, 3,  cat1["code"],  cat2["code"])

    conn.commit()

conn.close()