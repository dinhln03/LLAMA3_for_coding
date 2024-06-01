# -*- coding: utf-8 -*-

"""

Store data in the Sqlite3 Database

Table1



"""
import os
import sys
import codecs
import sqlite3

from common import log
from store.model import Question, Answer, Person, Topic

DB_PATH = 'spiderman.db'

logger = log.Logger(name='store')


def init_all_dbs():
    """
    call it when creating database
    :return:
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    exec_sql = "create table Question (id INTEGER primary key, content text, user VARCHAR(20), date VARCHAR(30))"
    cursor.execute(exec_sql)
    exec_sql = "create table People (id INTEGER primary key, question_id INTEGER, content text, user VARCHAR(20), date VARCHAR(30))"
    cursor.execute(exec_sql)
    conn.commit()
    conn.close()


def store_to_file(filename, question, answers):

    f = open(filename, 'w+')
    f.write(question)
    for ans in answers:
        f.write(ans)
    f.close()
    logger.info("Saved to file %s" % filename)


def store_new_question(question):

    assert isinstance(question, Question), "param `question` should be model.Question's instance"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = "INSERT into Question (id, content, user, date) VALUES (%s, %s, %s, %s)" % question()
    cursor.execute(sql)
    conn.commit()
    conn.close()


def init_people_file(directory):

    if directory[-1] != '/':
        directory += '/'
    path = directory + 'people.csv'

    if not os.path.exists(path):

        columns = [u'人物昵称', u'人物签名', u'人物标签', u'回答数', u'提问数', u'文章数', u'专栏数', u'想法数',
                   u'总赞同数', u'总感谢数', u'总收藏数', u'总编辑数', u'总关注数', u'被关注数', u'关注话题', u'关注专栏',
                   u'关注问题', u'收藏夹', u'动态']
        with codecs.open(path, 'a+', 'utf-8') as f:
            line = ','.join(columns)
            line += '\n'
            f.write(line)

        logger.info("Created people information file: %s" % path)
    return path


def init_question_file(directory):

    if directory[-1] != '/':
        directory += '/'
    path = directory + 'question.csv'

    if not os.path.exists(path):

        columns = [u'问题ID', u'问题标题', u'问题描述', u'问题关注数', u'问题浏览数', u'问题评论数', u'URL', u'回答文件']

        with codecs.open(path, 'a+', 'utf-8') as f:
            line = ','.join(columns)
            line += '\n'
            f.write(line)

        logger.info("Created question information file: %s" % path)
    return path


def save_file(content_type, content):

    if content_type == 'people':
        # 存储用户信息 -- csv
        dir_path = './result/people/'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        assert isinstance(content, Person), "use Person class instead of raw content"
        if content.name == '':
            return
        path = dir_path+'%s.txt' % content.name
        # path = init_people_file(dir_path)

        columns = [u'人物昵称', u'人物签名', u'人物标签', u'回答数', u'提问数', u'文章数', u'专栏数', u'想法数',
                   u'总赞同数', u'总感谢数', u'总收藏数', u'总编辑数', u'总关注数', u'被关注数', u'关注话题', u'关注专栏',
                   u'关注问题', u'收藏夹', u'动态']
        content = content.to_line()
        line = [u'[%s]:%s' % (x, y) for (x, y) in zip(columns, content)]
        line = '\n'.join(line)
        with codecs.open(path, 'a+', 'utf-8') as f:
            f.write(line)
        logger.info('people saved!')

    elif content_type == 'question':
        # 存储问题信息 -- csv
        dir_path = './result/question/'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        path = init_question_file(dir_path)
        # path = dir_path + 'question.txt'
        assert isinstance(content, Question), "use Question class instead of raw content"

        with codecs.open(path, 'a+', 'utf-8') as f:
            f.write(u'%s\n' % content.to_csv_line())
        logger.info('question saved')

    elif content_type == 'answers':

        """
        content format: {id:id/name, answers:[]}
        """

        assert isinstance(content, dict), "use Dict: {filename:xx, url:url, content: content, answers:[AnswerObjects]}"

        dir_path = './result/answers/'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        path = dir_path + content['filename']

        with codecs.open(path, 'a+', 'utf-8') as f:

            f.write(u'[Question]:%s\n' % content['content'])
            f.write(u'[URL]:%s\n\n' % content['url'])

            answers = content['answers']
            for ans in answers:
                f.write(u'%s\n' % ans)

        logger.info("answers saved")

    elif content_type == 'topic':

        assert isinstance(content, Topic), "use Tpoc class instead of raw content"
        dir_path = './result/topic/'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        path = dir_path + 'topic_%s.txt' % content.topic_id
        with codecs.open(path, 'a+', 'utf-8') as f:

            f.write(u'[标题]%s\n' % content.title)
            f.write(u'[类型]%s\n' % content.topic_type)

            for question in content.questions:

                f.write(u'[问题]:%s\n[回答作者]%s\n[回答内容]:%s\n[评论数]:%s\n\n' % question)

        logger.info('topic saved')

    else:
        pass
















