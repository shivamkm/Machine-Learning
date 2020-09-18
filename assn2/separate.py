import random
import sys

X_file = open("data.X","r")
Y_file = open("data.y","r")
train_file = open("train","w")
test_file = open("test","w")
split_ratio = float(sys.argv[1])

counter=0
for lineX in X_file:
    lineY = Y_file.readline()
    if counter == 0:
        counter = counter + 1
        continue
    else:
        linex = lineX.strip()
        liney = lineY.strip().split()
        output_str = ""
        for target in liney:
            item_number = target.split(':')[0]
            output_str = output_str + item_number+","
        output_str = output_str[:-1]+" "+linex+"\n"
        file_choice = random.random()
        if file_choice < split_ratio:
            train_file.write(output_str)
        else:
            test_file.write(output_str)

X_file.close()
Y_file.close()
train_file.close()
test_file.close()
