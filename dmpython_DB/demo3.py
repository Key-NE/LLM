#!/usr/bin/python
# coding:utf-8
import dmPython

try:
    # 连接数据库
    conn = dmPython.connect(user='SYSDBA', password='Dm123456', server='192.168.3.43', port=5236)
    cursor = conn.cursor()

    # 指定要查询的 SESSION_ID
    target_session_id = 'e10cb468-fb0e-4b30-86c7-8804b294cff0'

    try:
        # 查询数据
        query = "SELECT CONTENT, IS_AI, CREAT_TIME FROM HUICAI.CHAT_MESSAGE WHERE SESSION_ID = :1"
        cursor.execute(query, (target_session_id,))
        res = cursor.fetchall()

        # 打印查询结果
        for row in res:
            content, is_ai, create_time = row
            print(
                f"CONTENT: {content}, IS_AI: {is_ai}, CREATE_TIME: {create_time}")

        print('python: select success!')
    except (dmPython.Error, Exception) as err:
        print(f"查询数据时出错: {err}")

    # 关闭数据库连接
    conn.close()
except (dmPython.Error, Exception) as err:
    print(f"连接数据库时出错: {err}")
