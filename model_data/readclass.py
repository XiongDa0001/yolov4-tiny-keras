import xlwt  # 写入文件
import xlrd  # 打开excel文件

fopen = open('new_classes.txt', 'r')
lines = fopen.readlines()

i = 0
classes = []
for line in lines:
    line = line.strip('\n')  # 去掉换行符
    classes.append(line+'')
print(classes)
