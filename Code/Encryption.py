import random


def encrypt(numList):

    newList = []
    for entry in numList:
        key = random.uniform(0, entry/100)
        newList.append(entry + key)

    return newList


def decrypt(numList):

    newList = []
    for entry in numList:
        key = random.uniform(0, entry/100)
        newList.append(entry - key)

    return newList