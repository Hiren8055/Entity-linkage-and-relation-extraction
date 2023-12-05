import os
import time
from random import randint
import requests
from hashlib import md5
from bs4 import BeautifulSoup

"""TODO:
have to clean the data
bring it to pandas
"""

seed_list = []
visited = {}
frontier = {}
output_html = "htmls"

def read_seed_list():
    results = []
    with open("seedlist.txt") as feed:
        for row in feed.readlines():
            row = row.strip()
            results.append(row)
    return results

def extract_links_and_images(html_file):
    links = []
    images = []
    file_to_read = open(html_file)
    soup = BeautifulSoup(file_to_read.read())
    for link in soup.find_all("a"):
        candidate = link.attrs.get('href', '')
        if candidate.find("http") != -1 and candidate not in links:
            links.append(candidate)
    for img in soup.find_all("img"):
        candidate = img.attrs.get('src', '')
        if candidate.find("http") != -1 and candidate not in images:
            images.append(candidate)
    return links, images


def download_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        file_name = os.path.join(output_html, md5(url.encode()).hexdigest())
        file_to_save = open(file_name, "w")
        file_to_save.write(response.text)
        print(f"Saved file: {file_name}")
        return file_name
    else:
        print(f"Error while downloading {url}, status code: {response.status_code}")
    return ""

def update_frontier(new_links):
    for current in new_links:
        frontier[current] = ""

def crawl():
    for current in seed_list:
        frontier[current] = ""

    while len(frontier) > 0:
        time.sleep(randint(1, 10))
        url, _ = frontier.popitem()
        if url in visited:
            print(f"Skipping: {url} because its already crawled")
            continue
        saved_html = download_html(url)
        if len(saved_html) > 0:
            new_links, images = extract_links_and_images(saved_html)
            print(images)
            update_frontier(new_links)
        visited[url] = ""

if __name__ == "__main__":
    if not os.path.exists(output_html):
        os.mkdir(output_html)
    seed_list = read_seed_list()
    crawl()