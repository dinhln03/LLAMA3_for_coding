from flask import g
from app.comm.CompositeOperate import CompositeOperate
from app.comm.SqlExecute import SqlExecute
from app.module_config import table_module_map

class CommentController(CompositeOperate):
    def __init__(self, module):
        super(CommentController, self).__init__(module)

    def after_deal_get(self):
        comments = g.result.get("data")
        # 获取用户点赞记录
        user_id = g.flask_httpauth_user.get('id', None) if g.flask_httpauth_user else None
        # 点赞记录
        comment_licks_dict = dict()
        if user_id is not None:
            sql_query = table_module_map['bloglikelog'].sql_query_default
            sql_query = f'{sql_query} where bll_userid={user_id}'
            user_likes = SqlExecute.query_sql_data(sql_query)
            comment_licks_dict = {like['bll_blogcommentid']:like['bll_status'] for like in user_likes}
        # 所有评论根节点,添加用户是否点赞标志
        new_comments = []
        for comment in comments:
            comment['is_like'] = comment_licks_dict.get(comment['id']) or 0
            if not comment['bc_commentupid']:
                new_comments.append(comment)
        # new_comments = [comment for comment in comments if not comment['bc_commentupid']]
        for comment in new_comments:
            # 获取每个评论的回复
            comment['sub'] = [sub for sub in comments if sub['bc_commentupid']==comment['id']]
        g.result['data'] = new_comments

    def before_deal_post(self):
        g.json_data["data"]["bc_createuid"] = g.flask_httpauth_user.get('id')




