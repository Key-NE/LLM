import dmPython
conn=dmPython.connect(user='SYSDBA',password='Dm123456',server= '192.168.3.43',port=5236)
cursor = conn.cursor()
cursor.execute('select username from dba_users')
values = cursor.fetchall()
print(values)
cursor.close()
conn.close()