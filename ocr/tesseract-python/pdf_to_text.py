"""
# if this error found, please do below
export MAGICK_HOME="/usr/local/Cellar/imagemagick@6/6.9.9-34"

# output if page is not in correct orientation
python3.6 pdf_to_text.py
[":0:_0E00 000:00:0>> 05 :3:\n000g 000590; 05 :30\n:0:_0E00 mo0:0_>_ 0:0. :3:\n:0:_0E00 05 020030 00500 0:90\n\néoiodv'm'\n\n0:90 :0x000 0:: E0: 000:: 300:0; 30m>_>_ 0E :30\n”00 ___>> 03 “0:3 0_ 0:0: :0:0__0#0:_ 0E #30 00:00 0>> :0t<\n\n.0:n_ _0300m ”00.2\n05 :00 0Eo>00 >E H30 x00:0 .003 9 00:3 >>0:x 0.:00 30> = 05:00.: :30> :0 0_00__0>0 0_\nE26 30m 00:890. :30> #05 0:30 0x08 0000?: 00 30:83 0:: 0:230 ~:0__0 30m 0 003 __.0>>\n\n.00_0_> >E :_ 00000\n\n05 >>0__00 0:0 00E\\E0x000\\000300:0\\E00.:0x00_0.>>>>>>\\\\H00:: 9 om 0033 050:0”, 230:0\n30> H05 905% 0 “03.. 0.: .t “3000 0E3 9 :03E “0: 0_ 0:0,: .000_> 0E~ :_ :0:0__0H0:_ 00$.\n:0”, :0x000 0200: 0E 000 :00 30> 00 .005. 30: 0 E0300 003.. _ 0030000 .>x03_ 0:0 0>> 00::\n05:09: 50> :0 0:6:0 :0x00n. 0E __00‘0:_ 8 000: 30> .0E0:00 0>000 05 >28 0“ :090 :_\n\n.30000 >_:00:0\n:0 E._ .:00>V 0E x00m00_>_ >>0: >:_:0 >E :0 0_09 >00 >000E 0S __0 __0H0:_ 9 000: 0:00 _ .\n.0N00E 0 0_ E0E>0_000 .\n.0:0_0:0> 000:00:0>> 030_:0> v6 :0:_0E00 E0an_0>00 0 03 0:5 >__000 :00 _ .\n.0m0E_ :0x000 0800 50x0 05 003 0E0E:0:_>:0 E0E00_0>00 0:0 :0203005 >E .\n.0:_:00E >E :0 E0E:0:_>:0 E0E00_0>00 00000000 :03 0 .\n\n0:35:00 05 00 0:00:00 0005 .6 0E0m 050:0: 0_003_0> ”6 0_0300\n0 0E 0020 : 0030000 :0x000 :Es E0E00_0>00 _000_ 0x: _ .0:_:00E :30> :0 E0E:0:_>:0\n“:0E00_0>00 30m>_>_ + 000:00:0>> 0 03 “00 8 >>0: 30> :0000 9 0_ 92:0 0?: 00 000930 0:._.\n\n00:0_300:0:0 0:0 0::_"]


# save pdf file in correct orientation, you get correct output
python3.6 pdf_to_text.py
["Intro and prerequisites\n\nThe purpose of this article is to teach you how to set up a Wordpress + MySQL development\nenvironment on your machine. I like local development with Docker, because it gives me a\ncouple of valuable benefits. Some of these benefits are the following:\n\n0 a well separated development environment on my machine,\n\n0 my production and development environments use the exact same Docker image,\n\n0 I can easily bring up a development container of various Wordpress versions,\n\n0 deployment is a breeze,\n\n0 I don't need to install all the messy dev tools on my shiny new MacBook Pro (yeah, I’m an\norderly person).\n\nIn order to enjoy the above benefits, you need to install the Booker engine on your machine\nfirst. We are lucky, because I just bought a new Mac, so you can see the native Docker for\nMac installation in the Video. There is not much to write about it, it’s just a wizard that you\nshould follow. Just go to https://www.docker.com/products/docker#/mac and follow the\nsteps in my video.\n\nWe’ll use a SQL client during this tutorial, so please make sure that your preferred SOL client\nis available on your machine. If you don’t know what to use, Check out my favorite for the\nMac; Sequel Pro.\n\nAfter we sorted out the installation here is what we will do:\n\npull the MySQL (MariaDB) image from the Docker store\nstore data outside the container\n\nrun the MariaDB container\n\npull the Wordpress image\n\n@PPNT‘\n\nrun the Wordpress container"]


"""

import io
from PIL import Image
import pytesseract
from wand.image import Image as wi

pdf = wi(filename = "phototest.pdf", resolution = 300)
pdfImage = pdf.convert('jpeg')

imageBlobs = []

for img in pdfImage.sequence:
    imagePage = wi(image = img)
    imageBlobs.append(imagePage.make_blob('jpeg'))

extracted_text = []

for imageBlob in imageBlobs:
    image = Image.open(io.BytesIO(imageBlob))
    text = pytesseract.image_to_string(image, lang='eng')
    # text = pytesseract.image_to_string(image, lang='eng', boxes=False, \
    #                 config='--psm 7 --eom 3')
    extracted_text.append(text)

print(extracted_text)