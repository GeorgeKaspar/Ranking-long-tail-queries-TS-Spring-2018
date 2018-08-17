import numpy as np
import csv
from lang import *
from termcolor import colored

def process(query):
    query = query.replace(',', ' ').replace(':', ' ').replace('/', ' ').replace('\\', ' ').replace('\"', ' ').replace('\'', ' ')
    query = ' '.join(filter(lambda x: len(x) > 0, query.split()))
    print(colored(query, 'red'))
    query = change_keyboard_layout(query)
    print(colored(query, 'green'))
    return [query]

if __name__ == '__main__':
    with open("queries.tsv", "r") as q_file_from, open("queries_new.tsv", "w") as q_file_to:
        reader = csv.reader(q_file_from, delimiter='\t')
        writer = csv.writer(q_file_to, delimiter='\t')
        for row in reader:
            i = int(row[0])
            query = process(row[1])
            writer.writerow([row[0]] + query)