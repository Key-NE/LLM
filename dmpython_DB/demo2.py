#!/usr/bin/python
# coding:utf-8
import dmPython

try:
    # 连接数据库
    conn = dmPython.connect(user='SYSDBA', password='Dm123456', server='192.168.3.43', port=5236)
    cursor = conn.cursor()

    try:
        # 查询数据
        query = "SELECT ID, SESSION_ID, CONTENT, IS_AI, CREAT_TIME FROM HUICAI.CHAT_MESSAGE"
        cursor.execute(query)
        res = cursor.fetchall()

        # 打印查询结果
        for row in res:
            id_value, session_id, content, is_ai, create_time = row
            print(
                f"ID: {id_value}, SESSION_ID: {session_id}, CONTENT: {content}, IS_AI: {is_ai}, CREATE_TIME: {create_time}")

        print('python: select success!')
    except (dmPython.Error, Exception) as err:
        print(f"查询数据时出错: {err}")

    # 关闭数据库连接
    conn.close()
except (dmPython.Error, Exception) as err:
    print(f"连接数据库时出错: {err}")
