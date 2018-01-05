# import sys
#
# # print(sys.getrecursionlimit())   #1000   get current depth
#
# #factorial function with no recursion
# def factorial(number):
#     product = 1
#     for i in range(number):
#         product = product * (i + 1)
#     return product
#
# #resurvice function
# def factorial_recursive(number):
#     if number <= 1:
#         return 1
#     else:
#         return number*factorial(number -1)
#
# print(factorial(5))
# print(factorial_recursive(5))
#

#### binary tree recursive example
class tree():
    def __init__(self):
        self.Data = None
        self.Count = 0
        self.LeftSubtree = None
        self.RightSubtree = None

    def Insert(self, data):
        if self.Data == None:
            self.Data = data
            self.Count += 1
        elif data < self.Data:
            if self.LeftSubtree == None:
                self.LeftSubtree = tree()
            self.LeftSubtree.Insert(data)
        elif data == self.Data:
            self.Count += 1
        elif data > self.Data:
            if self.RightSubtree == None:
                self.RightSubtree = tree()
            self.RightSubtree.Insert(data)

if __name__ == '__main__':
    T = tree()
    T.Insert('b')
    T.Insert('a')
    T.Insert('c')
    print(T, T.Count, T.Data, T.RightSubtree.Data, T.LeftSubtree.Data)




