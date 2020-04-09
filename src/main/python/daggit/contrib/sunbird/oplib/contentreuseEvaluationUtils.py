import glob
import pandas as pd
import numpy as np   
import os
import json
import sys
import pyvips
import cv2
import img2pdf
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import natsort
from natsort import natsorted, ns


def concat_resize_min(image_ls):
    im1 = Image.open(image_ls[0])
    im2 = Image.open(image_ls[1])
    dst = Image.new('RGB', (im1.width + im2.width, im1.height+int(im2.height/2)) ) 
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def text_image(pdf_path, gocr_path, medium, Language_mapping_file, font_size, output_loc):
    MAX_W, MAX_H = 2000, 2000
    lang_mapping = json.loads(open(Language_mapping_file).read())
    font_path = lang_mapping[medium] 
    pages = convert_from_path(pdf_path, dpi= 500)
    for i in range(len(pages)): 
        try:
            gocr_img = os.path.join(output_loc , "Gocr_"+str(i)+'.png') 
            pdf_img = os.path.join( output_loc , "pdf_"+str(i)+ '.png') 
            pdf_page = pages[i]
            pdf_page.save(pdf_img, 'JPEG')
            
            with open(os.path.join(gocr_path ,"output-"+str(i+1)+"-to-"+str(i+1)+".json"), "r") as read_file:
                page_by_page_response = json.load(read_file)
            text = page_by_page_response['responses'][0]['fullTextAnnotation']['text']
            ls = [str(i) + ") " + text.splitlines()[i] for i in range(len(text.splitlines()))]
            joined_text = "\n".join(ls)
            chars_to_remove = ['&']
            joined_text = ''.join([c for c in joined_text if c not in set(chars_to_remove)])
            font = ImageFont.truetype(font_path, font_size)
            image = pyvips.Image.text(joined_text, width=MAX_W, height=MAX_H, fontfile = font_path,dpi=400) 
            image.write_to_file(gocr_img) 
            concat_resize_min([ pdf_img, gocr_img ]).save(os.path.join(output_loc,"output_"+str(i)+".png"))   
            os.remove(gocr_img)
            os.remove(pdf_img) 
        except:
            print("missed_pages"+str(i))