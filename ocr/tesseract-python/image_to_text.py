"""
Very basic python program extract text from image or pdf

Code from following link
https://github.com/nikhilkumarsingh/tesseract-python

pip3 install pytesseract pillow wand

"""


from PIL import Image
import pytesseract

image = Image.open("sample1.png")
text = pytesseract.image_to_string(image, lang="eng")
print(text)

