# This image picker only works for specific website layout. It needs to be modified to work with other websites!

# Load needed libraries
import requests
from bs4 import BeautifulSoup


# Specify url we are mining for images
url = ""

# Use get() method to retrieve the whole page and save the output into a variable
page = requests.get(url)

# Format page output with BeautifulSoup and save to a variable
soup = BeautifulSoup(page.text, 'html.parser')

# Loop through the page, append a website hostname and save all jpg image links
for src in soup.find_all('img'):
    img_link = url + src.get('src')     # get image links as string
    img_file = requests.get(img_link)       # open link above and save output to variable
    print(img_link)                         
    
    filename = src.get('alt') + ".jpg"  # I assume all of the images are jpgs, but I could be wrong
    
    file = open(filename, "wb")
    file.write(img_file.content)
    file.close

