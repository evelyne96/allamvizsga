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

def read_tsv(filename):
    modelText = []
    modelSentiment = []
    if os.path.exists(filename):
        with open(filename) as file:
            reader = csv.reader(file,  delimiter='\t')
            for row in reader:
                if row[3] == 0 or row[3] == 1:
                    modelSentiment.append(0)
                    modelText.append(row[2])
                else:
                    if modelSentiment != 2:
                        modelSentiment.append(1)
                        modelText.append(row[2])
    return modelText, modelSentiment

def write_to(filename,content):
    file = open(filename, "w")
    file.write(str(content))
    file.close()