
PATH = "D:\python_projects\\3.2\MWDistinguisher\datasets\\train\\targets.tsv"
class tswParser:
    def __init__(self, path:str):
        self.file = open(path, 'r')
    
    def ConvertIntoTxt(self, txtPath):
        txtData = open(txtPath, 'w')
        names_array = [line for line in self.file]
        names_array.sort()
        for name in names_array:
            txtData.write(name)
        txtData.close()

    def get_men_women(self):
        file = open('targets.txt', 'r')
        targets = [int(line.split('\t')[1][0]) for line in file]
        file.close()
        file = open('targets.txt', 'r')
        men = open('men.txt', 'w')
        women = open('women.txt', 'w')
        i = 0
        for line in file:
            if targets[i]==0:
                men.write(line)
            else:
                women.write(line)
            i+=1

    