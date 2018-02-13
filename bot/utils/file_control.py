import os
import csv

def read_from(filename):
        if os.path.exists(filename):
            file = open(filename, "r")
        else:
            file = open(filename, "w")
        lines = []
        for line in file:
            lines.append(line)
        file.close()
        return lines

def append_to(filename,line):
    file = open(filename, "a+")
    file.write('\n'+line)
    file.close()

def read_from_csv(filename,model):
    modelList = []
    if os.path.exists(filename):
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                isCreated,created = model.create_new(row)
                if isCreated:
                    modelList.append(created)
    return modelList

def write_to(filename,content):
    file = open(filename, "w")
    file.write(str(content))
    file.close()