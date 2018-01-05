#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3

import sys
from sys import stderr
import re

##additional func to use print in python3, when sys.stderr.write() is deprecated
def print_err(*args, **kwargs):
    print(*args, file=stderr, **kwargs)

def extract_names(filename):
    names = []
    f = open(filename, 'rU')
    text = f.read()

    ##<caption><h2>Popularity in 2008</h2></caption>

    year_match = re.search((r'Popularity\sin\s(\d\d\d\d)'), text)  #\s = whitespace
    if not year_match:
        print_err('Could\'nt find the year')
        sys.exit(1)
    year = year_match.group(1)
    names.append(year)

    ###<tr align="right"><td>9</td><td>Christopher</td><td>Elizabeth</td>

    tuples = re.findall(r'<td>(\d+)</td><td>(\w+)</td>\<td>(\w+)</td>', text)

    names_to_rank = {}
    for rank_tuple in tuples:
        (rank, boyname, girlname) = rank_tuple  #unpack the tuple into 3 vars
        if boyname not in names_to_rank:
            names_to_rank[boyname] = rank
        if girlname not in names_to_rank:
            names_to_rank[girlname] = rank

    sorted_names = sorted(names_to_rank.keys())

    for name in sorted_names:
        names.append(name + " " + names_to_rank[name])

    return names

def main():
    args = sys.argv[1:]

    if not args:
        print("usage: [--summaryfile] file [file...]")
        sys.exit(1)

    summary = False
    if args[0] == '--summaryfile':
        summary = True
        del args[0]

    for filename in args:
        names = extract_names(filename)
        text = '\n'.join(names)
        if summary:
            outf = open(filename + '.summary', 'w')
            outf.write(text + '\n')
            outf.close()
        else:
            print(text)

if __name__ == '__main__':
    main()
