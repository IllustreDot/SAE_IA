import random as rd

def data_One_Dimension():
    data = []
    for i in range(100):
        data.append(rd.randint(-100, 100))
    return data

def data_Two_Dimension():
    data=[]
    for i in range(100):
        data.append([0,0])
        for j in range(2):
            data[i][j]=rd.randint(-100, 100)
    return data