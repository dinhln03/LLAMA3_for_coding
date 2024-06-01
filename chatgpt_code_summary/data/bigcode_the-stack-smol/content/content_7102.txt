

from handlers.kbeServer.Editor.Interface import interface_workmark
from methods.DBManager import DBManager

#点赞
def Transactions_Code_2008( self_uid , self_username , json_data):

    # 回调json
    json_back = {
        "code": 0,
        "msg": "",
        "pam": ""
    }

    # json_data 结构
    uid = int(json_data["uid"])
    wid = int(json_data["wid"])
    lid = int(json_data["lid"])
    siscid = json_data["siscid"]
    tid = int(json_data["tid"])
    dian = int(json_data["dian"])
    ctype = int(json_data["ctype"])
    #uid ,wid ,lid ,siscid ,tid ,dian ,ctype

    # 获取下db的句柄，如果需要操作数据库的话
    DB = DBManager()
    json_back["code"] = interface_workmark.DoMsg_Dianzan(DB,self_uid,uid,wid,lid ,siscid ,tid ,dian ,ctype)
    DB.destroy()
    return json_back


#评分/评论
def Transactions_Code_2009( self_uid , self_username , json_data):

    # 回调json
    json_back = {
        "code": 0,
        "msg": "",
        "pam": ""
    }

    # json_data 结构
    uid = int(json_data["uid"])
    s_wid = int(json_data["s_wid"])
    s_lid = int(json_data["s_lid"])
    log = json_data["log"]
    score = int(json_data["score"])
    P_UID = int(json_data["P_UID"])
    ctype = int(json_data["ctype"])
    #uid ,wid ,lid ,siscid ,tid ,dian ,ctype

    # 获取下db的句柄，如果需要操作数据库的话
    DB = DBManager()
    json_back["code"] = interface_workmark.DoWorkMark(DB,self_uid,s_wid,s_lid,uid,score,log,P_UID,ctype)
    DB.destroy()
    return json_back


#获取评分数据
def Transactions_Code_2010( self_uid , self_username , json_data):

    # 回调json
    json_back = {
        "code": 0,
        "msg": "",
        "pam": ""
    }

    # json_data 结构
    uid = int(json_data["uid"])
    wid = int(json_data["wid"])
    lid = int(json_data["lid"])
    sis_cid = json_data["sis_cid"]
    ctype = int(json_data["ctype"])
    #uid ,wid ,lid ,siscid ,tid ,dian ,ctype

    # 获取下db的句柄，如果需要操作数据库的话
    DB = DBManager()
    json_back["code"] = 1
    json_back["pam"] = interface_workmark.DoWorkScoreData(DB,self_uid,wid,lid,uid,sis_cid,ctype)
    DB.destroy()
    return json_back

#获取评论数据
def Transactions_Code_2011( self_uid , self_username , json_data):

    # 回调json
    json_back = {
        "code": 0,
        "msg": "",
        "pam": ""
    }

    # json_data 结构
    uid = int(json_data["uid"])
    wid = int(json_data["wid"])
    lid = int(json_data["lid"])
    sis_cid = json_data["sis_cid"]
    ctype = int(json_data["ctype"])
    PID = int(json_data["PID"])
    ipage = int(json_data["ipage"])
    ilenght = int(json_data["ilenght"])

    # 获取下db的句柄，如果需要操作数据库的话
    DB = DBManager()
    json_back["code"] = 1
    json_back["pam"] = interface_workmark.DoWorkLogData(DB,self_uid,wid,lid, uid, sis_cid, PID , ipage , ilenght,ctype)
    DB.destroy()
    return json_back