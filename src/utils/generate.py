import random as rd

def data_One_Dimension(n):
    data = []
    for _ in range(n):
        data.append(rd.randint(-100, 100))
    return data

def data_Two_Dimension(n):
    data = []
    for _ in range(n):
        data.append([rd.randint(-100, 100), rd.randint(-100, 100)])
    return data