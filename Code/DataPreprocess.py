import pandas
import numpy


def open_0():

    csv_data = pandas.read_csv('Datasets/MARUTI_data.csv')
    csv_data = csv_data.dropna()

    row = csv_data.iloc[0]
    ave0 = []
    ave0.append(row['open'])
    ave0.append(row['high'])
    ave0.append(row['low'])
    ave0.append(row['adj_close'])
    ave0.append(row['volume'])
    ave0.append(row['close'])

    return csv_data, ave0


def read_0(csvData, ave0, num):

    row = csvData.iloc[num]

    line = []
    line.append(row['open']/ave0[0])
    line.append(row['high']/ave0[1])
    line.append(row['low']/ave0[2])
    line.append(row['adj_close']/ave0[3])
    line.append(row['volume']/ave0[4])
    line.append(row['close']/ave0[5])

    return line


def open_1():

    csv_data = pandas.read_csv('Datasets/real_estate_db.csv', encoding='ISO-8859-1')

    row = csv_data.iloc[0]
    ave1 = []
    for element in row:
        ave1.append(element)
    ave1 = ave1[13:]
    for i in range(len(ave1)):
        ave1[i] = ave1[i].item()

    return csv_data, ave1


def read_1(csvData, ave1, num):

    row = csvData.iloc[num]

    line = []
    for element in row:
        line.append(element)
    line = line[13:]

    for element in line:
        if numpy.isnan(element):
            return read_1(csvData, ave1, num+1)

    for i in range(len(line)):
        line[i] = line[i].item()/ave1[i]

    return line