import signal
import sys
import logging
import decimal
import datetime

from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)


def mysql_stream(conf, mongo, queue_out):
    logger = logging.getLogger(__name__)

    # server_id is your slave identifier, it should be unique.
    # set blocking to True if you want to block and wait for the next event at
    # the end of the stream

    # mysql 基础配置
    mysql_settings = {
        "host": conf['host'],
        "port": conf.getint('port'),
        "user": conf['user'],
        "passwd": conf['password']
    }

    res_dict = dict()
    dbs = conf['databases'].split(",")
    tables = ['already_table']

    for db in dbs:
        for table in tables:
            res_dict.update({table: mongo.get_log_pos(db, table)})

    for db in dbs:
        for table in tables:
            log_file, log_pos, resume_stream = res_dict.get(table)

            stream = BinLogStreamReader(connection_settings=mysql_settings,
                                        server_id=conf.getint('slaveid'),
                                        only_events=[DeleteRowsEvent, WriteRowsEvent, UpdateRowsEvent],
                                        blocking=True,
                                        resume_stream=resume_stream,
                                        log_file=log_file,
                                        log_pos=log_pos,
                                        only_tables=table,   # 只查询当前表的事件
                                        only_schemas=db)  # 只查询当前的数据库

            for binlogevent in stream:
                binlogevent.dump()
                schema = "%s" % binlogevent.schema
                table = "%s" % binlogevent.table

                # event_type, vals = None, None
                for row in binlogevent.rows:
                    if isinstance(binlogevent, DeleteRowsEvent):
                        vals = process_binlog_dict(row["values"])
                        event_type = 'delete'
                    elif isinstance(binlogevent, UpdateRowsEvent):
                        vals = dict()
                        vals["before"] = process_binlog_dict(row["before_values"])
                        vals["after"] = process_binlog_dict(row["after_values"])
                        event_type = 'update'
                    elif isinstance(binlogevent, WriteRowsEvent):
                        vals = process_binlog_dict(row["values"])
                        event_type = 'insert'

                    # 将事件类型和记录数值写入数据库中 将返回的 _id 当做序列号返回
                    seqnum = mongo.write_to_queue(event_type, vals, schema, table)

                    # 刷新 log 日志的位置
                    mongo.write_log_pos(stream.log_file, stream.log_pos, db, table)
                    # 将 _id 作为 seqnum 插入队列中
                    queue_out.put({'seqnum': seqnum})

                    logger.debug(f"------row------{row}")
                    logger.debug(f"------stream.log_pos------{stream.log_pos}")
                    logger.debug(f"------stream.log_file------{stream.log_file}")

    stream.close()


# 将非int数据全部转换为 str后续可能在导入的时候转换 则这里就是全部为 str
def process_binlog_dict(_dict):
    for k, v in _dict.items():
        if not isinstance(v, int):
            _dict.update({k: str(v)})
    return _dict

# # 因为是使用 csv 文档插入 所有不再对数据的类型做出校验
# def process_binlog_dict(_dict):
#     for k, v in _dict.items():
#         if isinstance(v, decimal.Decimal):
#             _dict.update({k: float(v)})
#         elif isinstance(v, datetime.timedelta):
#             _dict.update({k: str(v)})
#         elif isinstance(v, datetime.datetime):
#             _format = "%Y-%m-%d %H:%M:%S"
#             d1 = v.strftime(_format)
#             _new = datetime.datetime.strptime(d1, _format)
#             _dict.update({k: _new})
#
#     return _dict
