import sqlite3
from abc import ABCMeta, abstractmethod
from model.dao.daoexception import DAOException


class AbstractDAO(object):
    __metaclass__ = ABCMeta

    def __init__(self, conn):
        self._conn = conn

    """
    base CRUD operation
    """
    # GENERIC CREATE FUNCTION
    def _insert(self, request, parameters):
        with self._conn as conn:
            try:
                c = conn.cursor()
                c.execute(request, parameters)
                conn.commit()
                return c.lastrowid
            except sqlite3.Error as ex:
                conn.rollback()
                DAOException(self, ex)

    # GENERIC READ FUNCTION
    def _read(self, request, parameters=None):
        with self._conn as conn:
            try:
                c = conn.cursor()
                if parameters is None:
                    c.execute(request)
                else:
                    c.execute(request, parameters)
                return c.fetchall()
            except Exception as ex:
                DAOException(self, ex)

    # GENERIC UPDATE FUNCTION
    def _update(self, request, parameters):
        with self._conn as conn:
            try:
                c = conn.cursor()
                c.execute(request, parameters)
                conn.commit()
                return True
            except Exception as ex:
                conn.rollback()
                DAOException(self, ex)
                return False

    # GENERIC DELETE FUNCTION
    def _delete(self, request, obj_id):
        with self._conn as conn:
            try:
                c = conn.cursor()
                c.execute(request, obj_id)
                conn.commit()
                return True
            except Exception as ex:
                conn.rollback()
                DAOException(self, ex)
                return False

