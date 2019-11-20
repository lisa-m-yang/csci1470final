import tensorflow as tf
import numpy as np
import csv

# reads .csv files and converts them into text files

csv_file = raw_input('Enter the name of your input file: ')
txt_file = raw_input('Enter the name of your output file: ')

with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

