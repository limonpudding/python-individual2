# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re


def addFlag(res, num, sign, bytes):
    if re.search(sign, bytes):
        res[f"f{num}"] = 1
    else:
        res[f"f{num}"] = 0


def addFlagManySigns(res, num, signs, bytes):
    for entry in signs:
        if re.search(entry, bytes):
            res[f"f{num}"] = 1
    res[f"f{num}"] = 0


def checkFile(filePath):
    fileBytes = open(filePath, 'rb').read()
    result = {}

    # [1] for (i = 0; i < n; i++) (самый дефолтный цикл)
    addFlag(result, 1, b'\xc7.{1,9}\xeb.{1}\x8b.{1,5}\x83\xc0\x01\x89.{1,15}\x7d', fileBytes)

    # [2] sum += A[i] (сложение инта с элементом массива)
    addFlag(result, 2, b'\x8b\x85.{0,20}\xff\xff\x8b\x8d.{0,20}\xff\xff\x03\x8c\x85.{0,20}\xff\xff\x89\x8d.{0,20}\xff\xff', fileBytes)

    # [3] min > A[i] (сравнение инта с элементом массива)
    addFlag(result, 3, b'\x8b\x85.{0,2}\xff\xff\x8b.{0,11}\x3b.{0,6}(\x7d|\x7e)', fileBytes)

    # [4] min = A[i] (присвоение инту элемента массива)
    addFlagManySigns(result, 4, [b'\x8b\x45\xec\x8b\x4d\x08\x8b.{2,3}\x89',
                                 b'\x8b\x85.{2}\xff\xff\x8b\x4c\x85\x9c\x89'], fileBytes)

    # [5] temp = A[j]; A[j] = A[j + 1]; A[j + 1] = temp; (свап элементов, часто при сортировках)
    addFlag(result, 5, b'\x8b\x45\xec\x8b\x4d\x08\x8b.{2,3}\x89\x55\xe0\x8b\x45\xec\x8b\x4d\x08\x8b\x55\xec\x8b\x75\x08\x8b\x14\x96\x89.{2,3}\x8b\x45\xec\x8b\x4d\x08\x8b\x55\xe0\x89', fileBytes)

    # [6] A[j] < A[min] (сравнение элементов массива)
    addFlag(result, 6, b'\x8b\x45\xec\x8b\x4d\x08\x8b\x55\xec\x8b\x75\x08\x8b.{2,3}\x3b', fileBytes)

    # [7] A[i] = A[min] (присвоение элемента массива элементу массива)
    addFlagManySigns(result, 7, [b'\x8b.{0,3}\xe0\xff\xff\x8b.{0,3}\xe0\xff\xff\x8b.{0,3}\xe0\xff\xff\x89.{0,3}\xe0\xff\xff',
                                 b'\x8b\x45\xec\x8b\x4d\x08\x8b\x55\xec\x8b\x75\x08\x8b\x14\x96\x89.{2,3}'], fileBytes)

    # [8] sum += i (сложение инта с интом; i++ не считается)
    addFlag(result, 8, b'\x8b\x45.{2}\x45.{2}\x45', fileBytes)

    # [9] pr *= i (умножение инта на инт)
    addFlag(result, 9, b'\x8b\x45.{1}\x0f\xaf\x45.{1}', fileBytes)

    # [10] sum = 0 (присвоение переменной константы)
    addFlag(result, 10, b'\xc7\x45.{0,2}\x00\x00\x00', fileBytes)

    # [11] Заполнение элемента случайным числом A[i] = rand()
    addFlag(result, 11, b'\x8b\x85.{0,20}\xff\xff\x8b\x8d.{0,20}\xff\xff\x8b\x94\x8d.{0,20}\xff\xff\x89\x94\x85.{0,20}\xff\xff', fileBytes)

    # [12] Чтение с консоли cin >> x
    addFlagManySigns(result, 12, [b'\x8d\x45.{1}\x50\x8b\x0d\x9c\x30.{1}\x00\x51\xe8.{1}\xa0\xff\xff\x83\xc4\x08',
                                  b'\x8b\x0d.{3}\x00\xff\x15.{3}\x00\x3b[\xf0-\xff]'], fileBytes)

    # [13] Вывод в консоль cout << x
    result["f13"] = 0 #addFlag(b'', fileBytes)

    # [14] Присваивание элементу массива инта A[i] =x
    addFlagManySigns(result, 14, [b'\x8b\x45\xec\x8b\x4d\x08\x8b\x55\xe0\x89',
                                  b'\x8b.{0,10}\xc7\x84.{1}\x4c\xf0\xff\xff.{1}\x00\x00\x00'], fileBytes)

    return result


def searchAllInFolder(path, save=True):
    sr = None
    all = []
    if save:
        sr = open("search_result.csv", 'w')
    for entry in os.listdir(path):
        filePath = f"{path}\\{entry}"
        if (os.path.isfile(filePath)):
            csv = checkFile(filePath)
            all.append(csv)
            print(f"Файл: {entry} CSV: {csv} \n")
    all = str(all).replace("'", "\"")
    if save:
        sr.write(f"{all}\n")
    return all



# searchAllInFolder("control", True)


