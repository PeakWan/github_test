from django.db import connection,transaction
class Tool():
    # 获取sql数据
    def getData(sql, params):
        cursor = connection.cursor()
        cursor.execute(sql, params) if params else cursor.execute(sql)
        return cursor.fetchall()

    # 元组转集合#eg (('2021-04-21', 51, 27), ('2021-04-22', 75, 31))  parseArgsList=['date','loginNum','registerNum']To
    # [{'date': '2021-04-21', 'loginNum': 51, 'registerNum': 27}, {'date': '2021-04-22', 'loginNum': 75, 'registerNum': 31}]
    def tupleToMapList(tuple, parseArgsList):
        result = []
        dict = {}
        for element in tuple:
            for index in range(len(element)):
                dict[parseArgsList[index]] = element[index]
            result.append(dict)
            dict = {}
        return result