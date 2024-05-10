# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:36:51 2023

@author: 12142
"""

import streamlit as st
import sqlite3
import pandas as pd
import face_train
import test_detection
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import datetime

def create_connection():
    try:
        conn = sqlite3.connect('mydata.db') 
        cur = conn.cursor() 
        print('创建学生/连接数据库')
    except:
        print('连接学生表失败/待创建学生表已经存在')
        return
    try:          
        cur.executescript('''CREATE TABLE IF NOT EXISTS STUDENTS_DATA
                     (学号     CHAR(50)    PRIMARY KEY  NOT NULL,
                      姓名                 CHAR(50)    NOT NULL,
                      性别                 CHAR(50)    NOT NULL,
                      年级                 CHAR(50)    NOT NULL,
                      班级                 TEXT        NOT NULL,
                      签到次数             integer   NOT NULL
                      );
                      
                      CREATE TABLE IF NOT EXISTS SIGNIN_DATA
                        (日期学号     CHAR(50)    PRIMARY KEY  NOT NULL,
                        姓名                 CHAR(50)    NOT NULL,
                        年级                 CHAR(50)    NOT NULL,
                        班级                 TEXT        NOT NULL
                        );''')
        print('学生表创建成功')
    except:
        print('已成功创建学生表')
    finally:
        conn.commit()
        return conn

# def create_connection1():
#     try:
#         conn1 = sqlite3.connect('signin_data.db')  
#         print('创建签到表/连接数据库')
#     except:
#         print('连接签到表失败/待创建签到表已经存在')
#         return
#     try:          
#         conn1.execute('''CREATE TABLE SIGNIN_DATA
#                      (日期     CHAR(50)    PRIMARY KEY  NOT NULL,
#                       学号                 CHAR(50)    NOT NULL,
#                       姓名                 CHAR(50)    NOT NULL,
#                       年级                 CHAR(50)    NOT NULL,
#                       班级                 TEXT        NOT NULL,
#                       );''')
#         print('签到表创建成功')
#     except:
#         print('已成功创建签到表')
#     finally:
#         return conn1
    
def insert1(conn, date, iD, name, grade, stClass):
    print(f"date = {date}")
    print(f"iD = {iD}")
    # sql = f"SELECT * FROM SIGNIN_DATA WHERE 日期学号='{date+'-'+iD}'"
    # result = conn.execute(sql)
    # if not result:
    sql = f"INSERT INTO SIGNIN_DATA (日期学号,姓名,年级,班级) VALUES " \
        f"('{date+'-'+iD}','{name}','{grade}','{stClass}')"
    conn.execute(sql)
    conn.commit()

def add_signin(conn, date, iD):
    sql = f"SELECT * FROM STUDENTS_DATA WHERE 学号='{iD}'"
    result = conn.execute(sql).fetchall()
    # print(result)
    name = result[0][1]
    gender = result[0][2]
    grade = result[0][3]
    stClass = result[0][4]
    
    sql = f"SELECT * FROM SIGNIN_DATA WHERE 日期学号='{date+'-'+iD}'"
    Result = conn.execute(sql).fetchall()

    if not Result:
        count = result[0][5]+1
        insert1(conn, date, iD, name, grade, stClass)
        update(conn, iD, iD, name, gender, grade, stClass, count)
    else:
        st.error("该学生当天已经签过到")

def show_signin(conn):
    sql = "SELECT * FROM SIGNIN_DATA"
    result = conn.execute(sql).fetchall()
    
    if result:
        columns = ['日期学号', '姓名', '年级', '班级']
        df = pd.DataFrame(result, columns=columns)
        st.dataframe(df)
        return True
    else:
        st.error('没有签到信息')
        return False

def show_single_signin(conn, date):
    sql = f"SELECT * FROM SIGNIN_DATA WHERE 日期学号 LIKE '{date}%'"
    result = conn.execute(sql).fetchall()

    if result:
        columns = ['日期学号', '姓名', '年级', '班级']
        df = pd.DataFrame(result, columns=columns)
        st.dataframe(df)
    else:
        st.error("当天没有签到")

def login_page(session_state):
    st.subheader("用户登录")
    username = st.text_input("用户名", placeholder="测试用户名:ClassMonday")
    password = st.text_input("密码", type="password", placeholder="测试密码:45")
    login_button = st.button("登录")

    if login_button:
        if authenticate(username, password):
            session_state.login = True
            st.success("登入成功！")
            st.write("欢迎进入主界面：")
        else:
            st.error("用户名或密码错误，请重试。")


def authenticate(username, password):
    if username == "ClassMonday" and password == "45":
        return True
    else:
        return False


def add_st(conn):
    iD = st.text_input('请输入学生学号：')
    name = st.text_input('请输入学生姓名：')
    gender = st.text_input('请输入学生性别：')
    grade = st.text_input('请输入学生年级：')
    stClass = st.text_input('请输入学生班级：')
    courseCount = st.text_input('请输入该学生签到次数：')
    try:
        if st.button('添加'):
            insert(conn, iD, name, gender, grade, stClass, courseCount)
            st.success('添加成功，记得录入人脸信息')
    except:
        st.error('该学生已添加')

def modify(conn):
    iD0 = st.text_input('请输入要更新的学生信息：')
    sql = f"SELECT * FROM STUDENTS_DATA WHERE 学号='{iD0}'"
    result = conn.execute(sql).fetchall()

    if result:
        iD1 = st.text_input('请输入修改后的学生学号')
        name1 = st.text_input('请输入修改后的学生姓名：')
        gender1 = st.text_input('请输入修改后的学生性别：')
        grade1 = st.text_input('请输入修改后的学生年级：')
        stClass1 = st.text_input('请输入修改后的学生班级：')
        courseCountl = st.text_input('请输入修改后的学生签到次数：')

        if st.button('修改'):
            update(conn, iD0, iD1, name1, gender1, grade1, stClass1, courseCountl)
            st.success('修改成功')
    else:
        st.error('未找到该学生')

def inquiry(conn):
    iD0 = st.text_input('请输入要查询的学生学号：')
    sql = f"SELECT * FROM STUDENTS_DATA WHERE 学号='{iD0}'"
    result = conn.execute(sql).fetchall()

    if result:
        st.write('学号：', result[0][0])
        st.write('姓名：', result[0][1])
        st.write('性别：', result[0][2])
        st.write('年级：', result[0][3])
        st.write('班级：', result[0][4])
        st.write('签到次数：', result[0][5])
    else:
        st.error('未找到该学生')

def delete(conn):
    iD0 = st.text_input('请输入要删除的学生学号：')
    sql = f"SELECT * FROM STUDENTS_DATA WHERE 学号='{iD0}'"
    result = conn.execute(sql).fetchall()

    if result:
        if st.button('删除'):
            delete_row(conn, iD0)
            st.success('删除成功')
    else:
        st.error('未找到该学生')

def delete_all(conn):
    if st.button('清除所有学生数据'):
        sql = "DELETE FROM STUDENTS_DATA"
        conn.execute(sql)
        conn.commit()
        sql = "DELETE FROM SIGNIN_DATA"
        conn.execute(sql)
        conn.commit()
        st.success('清除成功')


def get_options(conn, column_name):
    sql = f"SELECT DISTINCT {column_name} FROM STUDENTS_DATA"
    result = conn.execute(sql).fetchall()
    options = [item[0] for item in result]
    return options


def get_selected_options(options, column_name):
    selected_options = []
    for option in options:
        if st.checkbox(option, key=f'{column_name}-{option}'):
            selected_options.append(option)
    return selected_options


def recommend_students(conn, selected_genders, selected_grades, selected_classes, selected_courseCounts):
    sql = f"SELECT * FROM STUDENTS_DATA WHERE 性别 IN ({','.join(['?']*len(selected_genders))}) AND 年级 IN ({','.join(['?']*len(selected_grades))}) AND 班级 IN ({','.join(['?']*len(selected_classes))}) AND 签到次数 IN ({','.join(['?']*len(selected_courseCounts))})"
    result = conn.execute(sql, [*selected_genders, *selected_grades, *selected_classes, *selected_courseCounts]).fetchall()
    
    if result:
        columns = ['学号', '姓名', '性别', '年级','班级','签到次数']
        df = pd.DataFrame(result, columns=columns)
        st.dataframe(df)
    else:
        st.error('未找到符合条件的学生')

def view_all_students(conn):
    sql = "SELECT * FROM STUDENTS_DATA"
    result = conn.execute(sql).fetchall()
    
    if result:
        columns = ['学号', '姓名', '性别', '年级', '班级', '签到次数']
        df = pd.DataFrame(result, columns=columns)
        st.dataframe(df)
    else:
        st.error('数据库中没有学生信息')

def insert(conn, iD, name, gender, grade, stClass, courseCount):
    sql = f"INSERT INTO STUDENTS_DATA (学号,姓名,性别,年级,班级,签到次数) VALUES " \
          f"('{iD}','{name}','{gender}','{grade}','{stClass}','{courseCount}')"
    conn.execute(sql)
    conn.commit()

def replace(conn, iD, name, gender, grade, stClass, courseCount):
    sql = f"REPLACE INTO STUDENTS_DATA (学号,姓名,性别,年级,班级,签到次数) VALUES " \
          f"('{iD}','{name}','{gender}','{grade}','{stClass}','{courseCount}')"
    conn.execute(sql)
    conn.commit()

def update(conn, iD0, iD1, name1, gender1, grade1, stClass1, courseCount1):
    sql = f"UPDATE STUDENTS_DATA SET 学号='{iD1}', 姓名='{name1}', 性别='{gender1}', " \
          f"年级='{grade1}', 班级='{stClass1}', 签到次数='{courseCount1}' " \
          f"WHERE 学号='{iD0}'"
    conn.execute(sql)
    conn.commit()

def delete_row(conn, iD):
    sql = f"DELETE FROM STUDENTS_DATA WHERE 学号='{iD}'"
    conn.execute(sql)
    conn.commit()

def excel_input(conn, excel_path):
    df = pd.read_excel(excel_path)
    try:
        for i in range(len(df)):
            iD, name, gender, grade, stClass, courseCount = df.loc[i, :]
            insert(conn, iD, name, gender, grade, stClass, courseCount)
        st.success("成功将初始数据导入数据库！(若初始数据文件已被导入则该数据将替换已被导入数据)")
    except:
        for i in range(len(df)):
            iD, name, gender, grade, stClass, courseCount = df.loc[i, :]
            replace(conn, iD, name, gender, grade, stClass, courseCount)
        st.success("初始数据文件已被导入则该数据将替换已被导入数据")



def camera_information():
    st.title("人脸信息录入")
    iD = st.text_input('请输入要采集学生的学号：')
    directory = "./data"
    test_detection.create_folder(directory, iD)

    test_detection.catch_face_info(iD, 400)

    iD_delete = st.text_input('请输入要删除学生人脸数据的学号')
    path_delete = f"{directory}/{iD_delete}"
    if st.button('删除指定的人脸数据'):
        test_detection.delete_folder(path_delete)


def camera_shot(conn):
    train_name = st.text_input("输入生成训练集模型的文件名")
    if st.button("用指定的图片集训练模型"):
        face_train.train(f'./model/{train_name}.keras')
        st.success(f"模型{train_name}已训练完成")
    keras_file = st.file_uploader("选择Keras模型文件", type="keras")
    if keras_file is not None:
        model_directory = "./model"
        model_path = f"{model_directory}/{keras_file.name}"
    
    test_detection.create_folder('./','tmp')
    test_detection.catch_face_info_re("tmp", 20)
    
    if st.button("开始拍照签到"):
        student_id = test_detection.face_test(model_path)
        #获取当前系统日期
        current_date = str(datetime.date.today())
        add_signin(conn, current_date, student_id)

        sql = f"SELECT * FROM STUDENTS_DATA WHERE 学号='{student_id}'"
        result = conn.execute(sql).fetchall()

        if result:
            st.write('日期：', current_date)
            st.write('学号：', result[0][0])
            st.write('姓名：', result[0][1])
            st.write('签到次数：', result[0][5])
        else:
            st.error('未找到该学生')
    sup_date = st.text_input("输入补充签到的日期以此格式 2024-05-10 ")
    sup_id = st.text_input("输入补充签到的学号")
    if st.button("补签"):
        if add_signin(conn, sup_date, sup_id):
            st.success("补签成功")
    
def main():
    # 创建或获取 session_state 对象
    session_state = st.session_state
    if "login" not in session_state:
        session_state.login = False

    # 如果未登录，显示登录页面
    if not session_state.login:
        login_page(session_state)
    else:
        st.title("课堂管家：学生面孔档案室")
        conn = create_connection()
        # conn1 = create_connection1()
        menu = ["导入Excel数据", "查询学生信息", "查询签到表", "进行拍照签到", "添加学生信息", "采集人脸信息", "修改学生信息", "删除学生信息", "清除所有数据"]
        choice = st.sidebar.selectbox("选择功能", menu)

        if   choice == "修改学生信息":
            modify(conn)
        elif choice == "删除学生信息":
            delete(conn)
        elif choice == "查询学生信息":
            inquiry(conn)
            gender_options = get_options(conn, '性别')
            grade_options = get_options(conn, '年级')
            class_options = get_options(conn, '班级')
            courseCount_options = get_options(conn, '签到次数')


            selected_genders = st.multiselect('性别' , gender_options)
            selected_grades = st.multiselect('年级' , grade_options)
            selected_classes = st.multiselect('班级', class_options)
            selected_courseCounts = st.multiselect('签到次数', courseCount_options)

            recommend_students(conn, selected_genders, selected_grades, selected_classes, selected_courseCounts)
            if st.button('查看所有学生'):
                view_all_students(conn)
        elif choice == "清除所有数据":
            delete_all(conn)
            
        elif choice == "添加学生信息":
            add_st(conn)

        elif choice == "导入Excel数据":
            excel_path = st.file_uploader("选择Excel文件", type="xlsx")
            if excel_path is not None:
                excel_input(conn, excel_path)

        elif choice == "采集人脸信息":
            camera_information()

        elif choice == "进行拍照签到":
            camera_shot(conn)
        elif choice == "查询签到表":
            Date = st.text_input("输入日期格式类似为: 2024-05-10")
            if st.button("查找"):
                show_single_signin(conn, Date)
            if st.button("显示完整签到表"):
                show_signin(conn)
        conn.close()

# def test_sign_table():
#     conn = create_connection()
#     excel_path = st.file_uploader("选择Excel文件", type="xlsx")
#     if excel_path is not None:
#         excel_input(conn, excel_path)
        
#     if st.button("插入数据"):
#         date = [
#             '2024-05-10','2024-05-11','2024-05-11','2024-05-11','2024-03-10'
#         ]
#         id = [
#             '2023303051052', '2023303051053', '2023303051054', '2023303051055', '2023303051056'
#         ]
#         for u, v in zip(date,id):
#             add_signin(conn, u, v)

#         show_signin(conn)
            
#     Date = st.text_input("输入日期")
#     if st.button("查找"):
#         show_single_signin(conn, Date)



if __name__ == '__main__':
    main()
    # test_sign_table()
