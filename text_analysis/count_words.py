"""
output

count_words.py
txt.txt has 252 words

"""

filename = 'txt.txt'
try:
    with open(filename) as file_object:
        contents = file_object.read()
except FileNotFoundError:
    message = "File not found"
    print(message)
else:
    words = contents.split()
    number_words = len(words)
    print(filename + " has " + str(number_words) +" words")