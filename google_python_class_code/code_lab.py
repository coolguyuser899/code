## youtube link https://www.youtube.com/watch?v=kWyoYtvJpe4

######### start of day 1 part 1

# a = 6
# a = 'Hello'
# a = "isn't"
# a = "I \"like\" this exercise"
# python string is immutable, once changed never change
# a = 'Hello'
# a.lower()
# a.find('e')   #find out how many occurance
# a[0] = 'H'    #find letter in string
# 'Hi %s I have %d donuts' % ('Alice', 42)   #% is place holder to replace value
# len(a)
# print(a, a, len(a))
# print(A)

# a = 'Hello'   # slicing
#  H  e  l  l  o
#  0  1  2  3  4
# -5 -4 -3 -2 -1
# a[0] # H
# a[1] # e
# a[1:3]  # el
# a[:]  # Hello
# a[1:]  #ello
# a[-4:-2]  # el
# a[-3:]  # llo

# 'Hello' + 6
# print('Hello' + 6)  #typeError
#
# #¡/usr/bin/python3 -tt     # -tt protect space indent of tab or spaces in code
#
# import sys
#
# def Hello(name):
#     if name == 'a' or name == 'b':
#         name = name + '????'
#     else:
#         print('Else')
#         #DoesNotExist(name)   #this line is skipped wehn input = a, python checks only at runtime, no compile
#     name = name + '!!!!'
#     print('Hello', name)   #comma or + works same
#
# def main():
#     # print('Hello')
#     Hello(sys.argv[1])   #output: ['/google_python_class_code/code_lab.py', 'a', 'b', 'c']
#
# if __name__ == '__main__':   #when other python module is loaded but not run it, this stmt is false
#     main()

# import sys
# dir(sys)
# help(sys.exit)
# help(len)
#
# len   #built-in function
# print(len('Hello'))

#python.org - documentation

######### end of day 1 part 1

######### start of day 1 part 2
#list is mutable, unlike string which is immutable, which assign variable a referene to another string when you see a='a, a='b'
# a = [1, 2, 3]
# a1 = [1, 2, 'a']
# a + a1  # ['1', '2', '3', '1', '2', 'a']
# len(a)  #3
# b = a  # b is not a copy of a, it points to same list
# a[0] = 13
# print(a)  #[13, 2, 3]
# print(b)  #[13, 2, 3]
# b = a[:]  #[13, 2, 3]
# a == b  #True
# a[1:3]  #[2, 3]
# a[:-1] #[13, 2]
#most often looping through list using following format
#for var in list:
#   print vars()
# for num in a: print(num) # one line format
# print(2 in a)   #return True
# a.append(4)
# a.pop(0)
# del a[1]
# print(a)
# a = [4, 3, 2, 1]
# print(sorted(a))   #built-in function 1,2,3,4, sorted make a new copy
# print(sorted(a, reverse=True))   #built-in function, 4,3,2,1
# a=sorted(a)
# a = ['aa', 'bb', 'ccc', 'dd']
# print(sorted(a, key=len))
# a = ['aa', 'bb', 'cccz', 'ddy']
# def Last(s): return s[-1]   #define func to sort last char of string
# print(sorted(a, key=Last))  #sort in order of last char ['aa', 'bb', 'ddy', 'cccz']
# print(':'.join(a))   #join string aa:bb:cccz:ddy
# print('\n'.join(a))   #print each value in separate single line
# b = ':'.join(a)
# print(b.split(':'))
# result = []
# for s in a: result.append(s)    #'aa', 'bb', 'ccc', 'dd' result
# range(20)   #generate range number
#####list size can change, tuple is fixed size
# a = (1, 2, 3)  #tuple is immutable, fixed size
# a[0] = 13    #TypeError: 'tuple' object does not support item assignment
# a = [(1, "b"), (2, "a"), (1, "a")]
# print(sorted(a))   #sorted first field first,then second
# (x, y) = (1, 2)   #parallel assignment, x=1, y=2
######### end of day 1 part 2

######### start of day 1 part 3
##hashtable, or map, dictionary delimiter is {}, key / value retreval is instant, one operation
##often map takes in incoherent data, and make it choerent, and organize them
##google runs by big hashtable by word, very powerful and useful
# d = {}
# d['a'] = 'alpha'
# d['o'] = 'omega'
# d['g'] = 'gamma'
# print(d.get('a'), d.get('x'))   #alpha None,  'None' returned if no key/value
# 'a' in d  #True
# 'x' in d  #False
# d.keys()    #return key list
# d.values()   #return all values
# for k in sorted(d.keys()): print('key:', k, '->', d[k])
# for tuple in d.items(): print(tuple)

####file handlinig
# import sys
# def Cat(filename):
#     f = open(filename, 'rU')    ##U takes care of dos or unix end of line char
#     ####option 1, the virtue of f is using small memory to read line by line
#     ##for line in f:
#     ##    print(line, end=" ")    ##python3 user end to change line ending
#
#     ####option 2, read content all at once, requires ram
#     # lines = f.readlines()
#     # print(lines)
#
#     ####option 3, read content as a string, useful to replace string or value
#     text = f.read()
#     print(text)
#
#     f.close()
#
# def main():
#     Cat(sys.argv[1])
# if __name__ == '__main__':
#     main()
# #
# import sys
# import operator
# from collections import Counter
# def countword(filename):
#     d = {}
#     file = open(filename, 'rU')
#
#     for word in file.read().split():
#         if word not in d:
#             d[word] = 1
#         else:
#             d[word] += 1
#     ###optino 1
#     ## for k, v in sorted(d.items(), key=lambda value: value[1], reverse=True):
#     # ###optino 2, use operator
#     for k, v in sorted(d.items(), key=operator.itemgetter(1), reverse=True):
#         print(k,v)
#     # # ###optino 3, use Counter
#     # c = Counter(d)
#     # print(c.most_common())
#
#     file.close()
#
# def main():
#     countword(sys.argv[1])
# if __name__ == '__main__':
#     main()
######### end of day 1 part 3

######### start of day 2 part 1
# . (dot) any char, including punctuation
# \w word char, letter, digit, _, space is NOT a \w
# \d digit
# \s whitespace,  \S non space char
# + 1 or more
# * 0 or more
#
import re   #search for patter in text
# # match = re.search('iig', 'called piiig')
# # print(match.group())   ##if not found, return None
def Find(pat, text):
    match = re.search(pat, text)
    if match: print(match.group())
    else: print('not found')
#
# #Find('iigs', 'called piiig')   #return when first occurance found
# #Find('...g', 'called piiig')    #any 3 chars and g, search from left to righ
# #Find('iig', 'called piiig, more piiig')   #return when first occurance found
# # Find('c\.l', 'c.lled piiig, more piiig')   # add \
# Find(r'c.l', 'c.lled piiig, more piiig')   #r is raw data no any special processing
# Find(r':\w\w\w', 'blah :cat')   #r is raw data no any special processing, : followed by 3 chars
# Find(r':\d\d\d', 'blah :cat :123')   #r is raw data no any special processing, : followed by 3 chars
# Find(r':\d\s\d\s\d', 'blah :cat :1 2 3')   #r is raw data no any special processing, : followed by 3 chars
# Find(r':\d\s+\d\s+\d', 'blah :cat :1     2     3')   # + one or more space
# Find(r':\w+', 'blah :cat&& :1     2     3')   # :cat
# Find(r':.+', 'blah :cat&& :1     2     3')   # :cat&& :1     2     3
# Find(r':\S+', 'blah :cat&&a=123&123 :1     2     3')   # non space char, :cat&&a=123&123, email address
#Find(r'\w+\.+\w+@\w+', 'blah firstname.lastname@gmail.com abc :1     2     3 @')   # non space char, :cat&&a=123&123, email address
# Find(r'[\w.]+@\w+', 'blah firstname.lastname@gmail.com abc :1     2     3 @')   # [] used for a set of chars, firstname.lastname@gmail
# Find(r'[\w.]+@[\w.]+', 'blah firstname.lastname@gmail.com abc :1     2     3 @')   # firstname.lastname@gmail.com
# Find(r'\w[\w.]+@[\w.]+', 'blah firstname.lastname@gmail.com abc :1     2     3 @')   # \w must start with a char
# m = re.search(r'([\w.]+)@([\w.]+)', 'blah first.last@gmail.com yatta @')    #separate in group in parenthesis ( )
# print(m, ' / ', m.group(), ' / ', m.group(1), ' / ', m.group(2))
#### re.search is 2nd favorite func, most used is findall() find all matches
# m = re.findall(r'[\w.]+@[\w.]+', 'blah first.last@gmail.com foo@bar')    #separate in group in parenthesis ( )
# print(m)   #['first.last@gmail.com', 'foo@bar']
# m = re.findall(r'[\w.]+@[\w.]+', open('code_lab.py', 'rU').read())    #find all matches in a file, useful
# print(m)   #['first.last@gmail.com', 'foo@bar']
#
# m = re.findall(r'([\w.]+)@([\w.]+)', 'blah first.last@gmail.com foo@bar')    #find all return tuples when use parenthesis()
# print(m)   #output [('first.last', 'gmail.com'), ('foo', 'bar')]

# m = re.findall(r'([\w.]+)@([\w.]+)', 'blah first.last@gmail.com foo@bar', re.IGNORECASE)    #use IGNORECASE flag
# print(m)   #output [('first.last', 'gmail.com'), ('foo', 'bar')]

