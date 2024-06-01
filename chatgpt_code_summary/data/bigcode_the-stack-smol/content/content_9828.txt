__author__ = 'miguel.freitas@checkmarx.com'

import os
import sys
import argparse
import pyodbc
import json
import array

DB = "CxDB"


def is_str(string):
    return string is not None and isinstance(string, str) and len(string) > 0


def is_int(integer):
    return not isinstance(integer, bool) and isinstance(integer, int)


def is_conn(conn):
    return conn is not None and isinstance(conn, pyodbc.Connection)


def read_file(filename):
    if is_str(filename):
        if filename.endswith(".json"):
            try:
                filename = os.path.basename(filename)
                if os.path.isfile(filename):
                    if os.access(filename, os.R_OK):
                        with open(filename, 'rb') as f:
                            return json.load(f)
                    else:
                        raise PermissionError("You don't have \
                            permissions to access this file")
                else:
                    raise FileNotFoundError("File Not Found")
            except FileNotFoundError:
                raise FileNotFoundError("File Not Found")
        else:
            raise AttributeError("File should have \".json\" extension")
    else:
        raise AttributeError("No Filename provided")


def connect_to_db(driver, server, user, password, database):
    if is_str(driver) and \
            is_str(server) and \
            is_str(user) and \
            is_str(password) and \
            is_str(database):
        try:
            conn = pyodbc.connect(
                'DRIVER={' + driver + '};SERVER=' + server +
                ';DATABASE=' + database +
                ';UID=' + user +
                ';PWD=' + password,
                timeout=3)
            print("Connection to", database, "success")
            return conn
        except pyodbc.OperationalError or \
                pyodbc.InterfaceError or \
                pyodbc.Error as error:
            raise ConnectionError(error)
    else:
        raise AttributeError(
            "server | user | password | database were not provided")


def get_category_type_id_by_name(conn, category_type_name):
    if is_conn(conn) and is_str(category_type_name):
        cursor = conn.cursor()
        category_type_id = -1
        cursor.execute(
            "SELECT id,Typename FROM dbo.CategoriesTypes WHERE TypeName=?",
            category_type_name)
        rows = cursor.fetchall()
        if len(rows) > 0:
            for row in rows:
                category_type_id = row[0]
                return category_type_id
        else:
            return category_type_id
    else:
        raise AttributeError(
            "Connection object or Category Name \
                was not provided")


def add_category_type_by_name(conn, category_type_name):
    if is_conn(conn) and is_str(category_type_name):
        cursor = conn.cursor()
        cursor.execute("SET IDENTITY_INSERT dbo.CategoriesTypes ON")
        conn.commit()
        cursor.execute(
            "INSERT INTO dbo.CategoriesTypes (Id, Typename) \
                VALUES((SELECT max(Id)+1 FROM dbo.CategoriesTypes), ?)",
            category_type_name)
        conn.commit()
        cursor.execute("SET IDENTITY_INSERT dbo.CategoriesTypes OFF")
        conn.commit()
        return True
    else:
        raise AttributeError(
            "Connection object or Category Name \
                was not provided")


def check_category_type_by_name(conn, category_type_name):
    if is_conn(conn) and is_str(category_type_name):
        category_type_id = get_category_type_id_by_name(
            conn, category_type_name)
        if category_type_id == -1:
            print("Category Type ", category_type_name, " does not exist.")
            add_category_type_by_name(conn, category_type_name)
            category_type_id = get_category_type_id_by_name(
                conn, category_type_name)
            print("Creating category type :",
                  category_type_name, "- ID:", category_type_id)
        else:
            print("Category already exists :",
                  category_type_name, "- ID:", category_type_id)
        return category_type_id
    else:
        raise AttributeError(
            "Connection object or Category Name \
                was not provided")


def delete_categories_by_category_type_id(conn, category_type_id):
    if is_conn(conn) and is_int(category_type_id):
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM dbo.Categories WHERE CategoryType=?",
            category_type_id)
        conn.commit()
    else:
        raise AttributeError(
            "Connection object or Category Type ID \
                was not provided")


def delete_categories_for_queries_by_category_type_id(conn, category_type_id):
    if is_conn(conn) and is_int(category_type_id):
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM dbo.CategoryForQuery WHERE CategoryId \
                IN (SELECT id FROM dbo.Categories WHERE CategoryType=?)",
            category_type_id)
        conn.commit()
    else:
        raise AttributeError(
            "Connection object or Category Type ID \
                was not provided")


def clean_old_data(conn, category_type_id):
    if is_conn(conn) and is_int(category_type_id):
        delete_categories_for_queries_by_category_type_id(
            conn, category_type_id)
        delete_categories_by_category_type_id(conn, category_type_id)
        print("Clearing old data...")
    else:
        raise AttributeError(
            "Connection object or Category Type ID \
                was not provided")


def add_category(conn, category_name, category_type_id):
    if is_conn(conn) and is_str(category_name) and is_int(category_type_id):
        cursor = conn.cursor()
        cursor.execute("SET IDENTITY_INSERT dbo.Categories ON")
        conn.commit()
        cursor.execute("INSERT INTO dbo.Categories (Id, CategoryName,CategoryType) \
            VALUES((SELECT max(Id)+1 FROM dbo.Categories),?,?)",
                       (category_name, category_type_id))
        conn.commit()
        cursor.execute("SET IDENTITY_INSERT dbo.Categories OFF")
        conn.commit()
        return True
    else:
        raise AttributeError(
            "Connection object or Category Name or Category Type ID \
                was not provided")


def get_category_id(conn, category_name, category_type_id):
    if is_conn(conn) and is_str(category_name) and is_int(category_type_id):
        cursor = conn.cursor()
        cursor.execute("SELECT Id FROM dbo.Categories WHERE \
            CategoryName=? AND CategoryType=?",
                       (category_name, category_type_id))
        return cursor.fetchall()[0][0]
    else:
        raise AttributeError(
            "Connection object or Category Name or Category Type ID \
                was not provided")


def add_category_for_query(conn, category_id, query_id):
    if is_conn(conn) and is_int(category_id) and is_int(query_id):
        cursor = conn.cursor()
        cursor.execute("SET IDENTITY_INSERT dbo.CategoryForQuery ON")
        conn.commit()
        cursor.execute(
            "INSERT INTO dbo.CategoryForQuery (Id,QueryId,CategoryId) \
                VALUES((SELECT max(Id)+1 FROM dbo.CategoryForQuery),?,?)",
            (query_id, category_id))
        conn.commit()
        cursor.execute("SET IDENTITY_INSERT dbo.CategoryForQuery OFF")
        conn.commit()
        return True
    else:
        raise AttributeError(
            "Connection object or Category ID or Query ID \
                was not provided")


def get_categories_by_category_type_id_and_name(conn,
                                                category_name,
                                                category_type_id):
    if is_conn(conn) and is_int(category_type_id) and is_str(category_name):
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dbo.Categories WHERE \
            CategoryName=? AND CategoryType=?",
                       category_name, category_type_id)
        rows = cursor.fetchall()
        return rows
    else:
        raise AttributeError(
            "Connection object or Category ID or Query ID \
                was not provided")


def insert_new_categories(conn, category_type_id, group):
    if is_conn(conn) and is_int(category_type_id):
        if "name" in group:
            category_name = group["name"]
            add_category(conn, category_name, category_type_id)
            category = get_categories_by_category_type_id_and_name(
                conn, category_name, category_type_id)
            print("\nNew Category Inserted : ", category[0])
            category_id = category[0][0]
            return category_id
    else:
        raise AttributeError(
            "Connection object or Category Type ID \
                was not provided")


def get_queries(conn, query_ids_list):
    if is_conn(conn) and len(query_ids_list) > 0:
        sanitized_list = []
        for queryId in query_ids_list:
            if is_int(queryId):
                sanitized_list.append(queryId)
        query_ids = str(sanitized_list).strip('[]')
        if len(query_ids) > 0:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM dbo.Query WHERE QueryId IN (" + query_ids + ")")
            return cursor.fetchall()
    else:
        raise AttributeError("Connection object or Query List \
            was not provided")


def get_categories_ids_by_category_type(conn, category_type_id):
    if is_conn(conn) and is_int(category_type_id):
        cursor = conn.cursor()
        cursor.execute(
            "SELECT [Id] FROM dbo.Categories where CategoryType=?",
            category_type_id)
        rows = cursor.fetchall()
        arr = array.array('i')
        for row in rows:
            category_id = row[0]
            arr.append(category_id)
        return arr

    else:
        raise AttributeError(
            "Connection object or Category Type ID \
                was not provided")


def insert_queries(conn, category_id, severity_id, queries):
    if is_conn(conn) and is_int(category_id) and \
            is_int(severity_id) and len(queries) > 0:
        cursor = conn.cursor()
        cursor.execute("SET IDENTITY_INSERT dbo.CategoryForQuery ON")
        conn.commit()
        i = 0
        for query in queries:
            query_id = query[0]
            percentage = round((i * 100) / len(queries), 0)
            print("Inserting Query", query_id, "...", percentage, "%")
            cursor.execute("INSERT INTO dbo.CategoryForQuery \
                (Id, QueryId,CategoryId) VALUES\
                ((SELECT max(Id)+1 FROM dbo.CategoryForQuery), ?,?)",
                           (query_id, category_id))
            conn.commit()
            update_customized_queries(conn, category_id, severity_id,
                                      query_id)
            i = i + 1
        cursor.execute("SET IDENTITY_INSERT dbo.CategoryForQuery OFF")
        conn.commit()
    else:
        raise AttributeError(
            "Connection object or Category ID \
                was not provided")


def get_args(args):
    if isinstance(args, list) and len(args) > 0:
        args_parser = argparse.ArgumentParser(
            description='Add Custom Category to CxDB')
        args_parser.add_argument(
            '-dbd', '--dbdriver', help='Checkmarx MSSQL DB Driver',
            required=False,
            default="SQL Server")
        args_parser.add_argument(
            '-dbu', '--dbuser', help='Checkmarx MSSQL DB Username',
            required=True)
        args_parser.add_argument('-dbp', '--dbpassword',
                                 help='Checkmarx MSSQL DB Password',
                                 required=True)
        args_parser.add_argument('-dbs', '--dbserver',
                                 help='Checkmarx MSSQL DB Server URL',
                                 required=True)
        args_parser.add_argument('-fg', '--file_groups',
                                 help='Categories and Queries Mapping File',
                                 required=True)
        return args_parser.parse_args(args)
    else:
        raise AttributeError("args should be a non-empty array")


def update_category_severity_mapping(conn, severity_id,
                                     category_name, group_name):
    if is_conn(conn) and is_int(severity_id) and is_str(category_name) and \
            is_str(group_name):
        cursor = conn.cursor()
        cursor.execute("UPDATE Queries \
            SET Queries.Severity = ? \
            FROM dbo.Categories Categories \
            JOIN dbo.CategoryForQuery CategoriesForQuery \
                ON Categories.Id=CategoriesForQuery.CategoryId \
            JOIN dbo.Query Queries \
                ON CategoriesForQuery.QueryId=Queries.QueryId \
            JOIN dbo.CategoriesTypes CategoriesTypes \
                ON Categories.CategoryType = CategoriesTypes.Id \
            WHERE Categories.CategoryName = ? \
            AND CategoriesTypes.TypeName = ?",
                       (severity_id, group_name, category_name))
        conn.commit()
        cursor.execute("UPDATE QueryVersions \
            SET QueryVersions.Severity = ? \
            FROM dbo.Categories Categories \
            JOIN dbo.CategoryForQuery CategoriesForQuery \
                ON Categories.Id=CategoriesForQuery.CategoryId \
            JOIN dbo.QueryVersion QueryVersions \
                ON CategoriesForQuery.QueryId=QueryVersions.QueryId \
            JOIN dbo.CategoriesTypes CategoriesTypes \
                ON Categories.CategoryType = CategoriesTypes.Id \
            WHERE Categories.CategoryName = ? \
            AND CategoriesTypes.TypeName = ?",
                       (severity_id, group_name, category_name))
        conn.commit()
        print("Updating Severity Mapping for Severity", severity_id,
              "-", group_name, "-", category_name)
    else:
        raise AttributeError(
            "Connection object was not provided")


def update_customized_queries(conn, category_id, severity_id, query_id):
    if is_conn(conn) and is_int(category_id) and is_int(severity_id) \
            and is_int(query_id):
        cursor = conn.cursor()

        cursor.execute("SELECT QueryId FROM dbo.Query \
                WHERE PackageId IN (\
            SELECT DISTINCT PackageId FROM dbo.QueryGroup \
                WHERE Name = (\
            SELECT Name FROM dbo.QueryGroup \
                WHERE PackageId = (\
            SELECT DISTINCT PackageId FROM dbo.Query \
                WHERE QueryId = ?)\
            ) and PackageId > 100000\
            ) and Name = (\
            SELECT DISTINCT Name FROM dbo.Query \
                WHERE QueryId = ?)",
                       (query_id, query_id))
        customized_queries_list = cursor.fetchall()
        if len(customized_queries_list) > 0:
            for customized_query in customized_queries_list:
                cursor.execute("INSERT INTO dbo.CategoryForQuery \
                    (Id, QueryId,CategoryId) VALUES\
                    ((SELECT max(Id)+1 FROM dbo.CategoryForQuery), ?,?)",
                               (customized_query[0], category_id))
                conn.commit()

            cursor.execute("UPDATE dbo.QueryVersion SET Severity = ? \
                WHERE QueryId IN (\
                SELECT QueryId FROM dbo.QueryVersion \
                    WHERE PackageId IN (\
                SELECT DISTINCT PackageId FROM dbo.QueryGroup \
                    WHERE Name = (\
                SELECT Name FROM dbo.QueryGroup \
                    WHERE PackageId = (\
                SELECT DISTINCT PackageId FROM dbo.QueryVersion \
                    WHERE QueryId = ?)\
                ) and PackageId > 100000\
                ) and Name = (\
                SELECT DISTINCT Name FROM dbo.QueryVersion \
                    WHERE QueryId = ?)\
                )",
                           (severity_id, query_id, query_id))
            conn.commit()
            cursor.execute("UPDATE dbo.Query SET Severity = ? \
                    WHERE QueryId IN (\
                SELECT QueryId FROM dbo.Query \
                    WHERE PackageId IN (\
                SELECT DISTINCT PackageId FROM dbo.QueryGroup \
                    WHERE Name = (\
                SELECT Name FROM dbo.QueryGroup \
                    WHERE PackageId = (\
                SELECT DISTINCT PackageId FROM dbo.Query \
                    WHERE QueryId = ?)\
                ) and PackageId > 100000\
                ) and Name = (\
                SELECT DISTINCT Name FROM dbo.Query \
                    WHERE QueryId = ?)\
                )",
                           (severity_id, query_id, query_id))
            conn.commit()
            print("Updating Customized Queries Severity", severity_id,
                  "- Query ID -", query_id)
        else:
            print("No existing customized queries for", query_id)
        return True
    else:
        raise AttributeError(
            "Connection object bwas not provided")


def main(args):
    if args is not None and hasattr(args, "file_groups"):
        file_groups = args.file_groups
        if is_str(file_groups):
            file_content = read_file(file_groups)
            category = file_content["category"]
            category_name = category["name"]
            groups = category["groups"]
            if hasattr(args, "dbdriver") and \
                    hasattr(args, "dbserver") and \
                    hasattr(args, "dbuser") and \
                    hasattr(args, "dbpassword"):
                db_server = args.dbserver
                db_user = args.dbuser
                db_pwd = args.dbpassword
                db_driver = args.dbdriver
                if is_str(db_driver) and \
                        is_str(db_server) and \
                        is_str(db_user) and \
                        is_str(db_pwd):

                    conn = connect_to_db(
                        db_driver, db_server, db_user, db_pwd, DB)

                    if is_conn(conn):
                        category_type_id = check_category_type_by_name(
                            conn, category_name)
                        clean_old_data(conn, category_type_id)
                        for group in groups:
                            category_id = insert_new_categories(
                                conn, category_type_id, group)
                            if "query_ids" in group and "name" in group and \
                                    "severity_id" in group:
                                severity_id = group["severity_id"]
                                group_name = group["name"]
                                queries = get_queries(conn, group["query_ids"])
                                print(group_name, ":", len(queries),
                                      "queries to change")
                                insert_queries(conn, category_id,
                                               severity_id, queries)
                                update_category_severity_mapping(
                                    conn,
                                    severity_id,
                                    category_name,
                                    group_name)
                            else:
                                print("Group has 1 missing attribute: name\
                                    query_ids or severity_id")

                    else:
                        raise Exception("Cannot connect to Database")
                else:
                    raise Exception(
                        "db_server | db_user | db_pwd \
                            are not valid strings")
            else:
                raise Exception(
                    "db_server | db_user | db_pwd \
                        was not provided as an argument")
        else:
            raise TypeError("file_groups is not a string")
    else:
        raise AttributeError("args does not has file_groups as attribute")


if __name__ == "__main__":
    main(get_args(sys.argv[1:]))
