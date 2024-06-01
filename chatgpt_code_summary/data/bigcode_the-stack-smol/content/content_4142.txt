# -*- coding: utf-8 -*-

import os
import sqlite3
import logging

logger = logging.getLogger("xtc")

class sqlite_handle(object):

    def __init__(self):
        self.dbname = "Xsense.db"
        self.conn = None

    def db_init(self):  # 初始化db task_info、apps、scripts、run_tasks
        self.db_table_all()
        conn = sqlite3.connect(self.dbname)
        try:
            for cre in self.create_dic:
                conn.execute(cre)
                # logger.info(cre)
        except Exception as e:
            logger.info("Create table failed: {}".format(e))
            return False
        finally:
            conn.close()

    def insert_task(self,taskdict):   # 插入任务信息 for
        conn = sqlite3.connect(self.dbname)
        for task in taskdict:
            conn.execute(
                'INSERT INTO task_Info VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',task
                )
        conn.commit()
        conn.close()

    def insert_script_one(self,scriptOne): # 插入脚本信息 
        conn = sqlite3.connect(self.dbname)
        conn.execute(
            'INSERT INTO scripts VALUES (?,?,?,?,?,?,?,?)',scriptOne
            )
        conn.commit()
        conn.close()

    def insert_task_many(self,script_data):   # 插入任务信息 多项
        conn = sqlite3.connect(self.dbname)
        conn.executemany(
                'INSERT INTO scripts VALUES (?,?,?,?,?,?,?,?)',script_data
            )
        conn.commit()
        conn.close()

    def db_table_all(self):
        crt_task_info = '''CREATE TABLE IF NOT EXISTS task_info (
                taskId INT, testTaskName TEXT, optType int,scriptId INT,scriptUrl TEXT,
                startDate int, endDate int, exeBeginTime TEXT, exeEndTime TEXT,
                exeType int, interval int, iterationNum int, startIterationNumber int
                );'''
        crt_scripts = '''CREATE TABLE IF NOT EXISTS scripts (
                scriptId INT, scriptName TEXT, scriptType int,scriptUrl TEXT,
                uploadDate int, scriptMaxRunTime int, scriptVersion int,
                scriptCacheUrl TEXT
                );'''
        crt_apps = '''CREATE TABLE IF NOT EXISTS apps (
                scriptId INT, appCheck int, appPackageName TEXT, appUrl TEXT, appMd5 TEXT,
                appVersion TEXT, appVersionCode TEXT, appLastUpdateTime TEXT, appCacheUrl TEXT
                );'''
        run_tasks = '''CREATE TABLE IF NOT EXISTS run_tasks (
                taskId INT, testTaskName TEXT, optType int,scriptId INT,scriptUrl TEXT,
                startDate int, endDate int, exeBeginTime TEXT, exeEndTime TEXT,
                exeType int, interval int, iterationNum int, startIterationNumber int
                );'''
        create_dic = []
        create_dic.append(crt_task_info)
        create_dic.append(crt_scripts)
        create_dic.append(crt_apps)
        create_dic.append(run_tasks)    # 保存需要运行的任务 有必要么
        self.create_dic = create_dic

    def query_runtask(self):
        conn = sqlite3.connect(self.dbname)
        taskrows = [] #元素为tuple，(205937, 'pyclient-test', 1, 107864, 'http://202.105.193....69910.zip', 20191006000000, 20201231235959, '000000', '235959', 2, 1, 1, 1)
        # 获取未完成的按次任务 不含重复项  新增+启动, exeType=2按次执行   exeType=1按时执行
        # optType  1`=新增任务；`2`=暂停任务；`3`=启动任务；`4`=删除任务
        for row in conn.execute('SELECT DISTINCT * FROM task_info WHERE optType=3 OR optType=1 AND exeType=2 AND startIterationNumber<=iterationNum'):
            taskrows.append(row)
        conn.close()
        return taskrows

    def dele_table(self):
        pass

    def query(self, sql, sqlstring=False):
        conn = sqlite3.connect(self.dbname)
        cursor = conn.cursor()
        # cursor = self.conn.cursor()
        if sqlstring:
            cursor.executemany(sql, sqlstring)
        else:
            cursor.execute(sql)
        data = cursor.fetchall()
        cursor.close()
        return data

    def update(self, sql, sqlstring=False):
        conn = sqlite3.connect(self.dbname)
        cursor = conn.cursor()
        # cursor = self.conn.cursor()
        if sqlstring:
            cursor.executemany(sql, sqlstring)
        else:
            cursor.execute(sql)
        conn.commit()
        cursor.close()

    def _update(self, sql, value=None, querymany=True):
        ret = True
        try:
            if querymany:
                self.update(sql, value)
            else:
                self.update(sql)
        #except SqliteException:
        except Exception as e:
            logger.info("error('执行sqlite: {} 时出错：{}')".format(sql, e))
            ret = False
        return ret

    def del_task_byid(self, taskid):
        conn = sqlite3.connect(self.dbname)
        cursor = conn.cursor()
        sql = 'DELETE FROM task_info WHERE taskid={}'.format(taskid)
        cursor.execute(sql)
        logger.info("刪除taskid={}  cursor.rowcount={}".format(taskid, str(cursor.rowcount)))
        conn.commit()
        cursor.close()
        conn.close()

    def update_task_run_status(self, taskid, status):
        conn = sqlite3.connect(self.dbname)
        cursor = conn.cursor()
        cursor.execute("UPDATE task_info SET optType={} WHERE taskid={}".format(status, taskid))
        logger.info("更新taskid={}，设置optType={}，cursor.rowcount={}".format(taskid, status, str(cursor.rowcount)))
        conn.commit()
        cursor.close()
        conn.close()

    def update_task_run_count(self, taskid, run_count):
        conn = sqlite3.connect(self.dbname)
        cursor = conn.cursor()
        cursor.execute("UPDATE task_info SET startIterationNumber={} WHERE taskid={}".format(run_count, taskid))
        logger.info("更新taskid={}，startIterationNumber={}，cursor.rowcount={}".format(taskid, run_count, str(cursor.rowcount)))
        conn.commit()
        cursor.close()
        conn.close()

    def updata_table(self):
        pass

if __name__ == "__main__":
    handle = sqlite_handle()
    if not os.path.isfile(handle.dbname):
        handle.db_init()
    #taskrows = handle.query_runtask()
    #print("taskrows=" + str(taskrows))
    #handle.del_task_byid("1235")
    handle.update_task_run_count("206266", 60)

    #handle.update_task_run_status("206266", "5")

    # 更新/删除 单条任务、更新 脚本信息
    # 下载前查询数据库，如果脚本id已经存在，且更新时间一致 则不下载，否则下载-->入库
    # 任务运行，先检查是否有新任务，如果有新任务，则入库，
    #     没有新任务，则查询数据库，任务id运行信息是否达到rm条件（过期、完成等）
    #         如果运行 轮次达到 总轮次 则del
    #         如果 结束时间超过当前时间  则del
    #     此处需要增加 id 排序 后再运行
    #     运行完成后，更新 id对应的轮次信息
    # 今天搞定 脚本运行和结果文件 ，然后做db update 和 remove