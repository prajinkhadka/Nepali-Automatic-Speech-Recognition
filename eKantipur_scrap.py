import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as BS
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# define a url, which will be scrapped 
url = "https://ekantipur.com/news"

# extract a category
http = urllib3.PoolManager()
http.addheaders = [('User-agent', 'Mozilla/61.0')]
web_page = http.request('GET', url)
soup = BS(web_page.data, 'html5lib')
ndict = {"Content":[]}


categories={"news":"https://ekantipur.com/news/",
            "business":"https://ekantipur.com/business/",
            "world":"https://ekantipur.com/world/",
            "sports":"https://ekantipur.com/sports"
          }
show=True

def scrap_news():
  for category, url in categories.items():
    print(category)
    print(url)
    web_page = http.request('GET', url)
    soup = BS(web_page.data, 'html5lib')

    # loop through all the divs with '.normal` class found in the webpage
    for row in soup.select(".normal"):
      # title is of h2 element
      title = row.find("h2")

      # description is on p element
      description = row.find("p").text
      # get title text
      title_text=title.text
      title_link=title.a.get("href")
      # print(title_link)
      if title_link.split(":")[0]!="https":
        title_link=url.split(f"/{category}")[0]+title.a.get("href")
      title_text=title.text
      #print(title_link)
      
      news_page = http.request('GET', title_link)
      news_soup = BS(news_page.data, 'html5lib')

      date = news_soup.find("time").text
      author_url = news_soup.select_one(".author").a.get("href")
      author_name = news_soup.select_one(".author").text
      news_content=""
      for content in news_soup.select_one(".row").findAll("p"):
        content = str(content).split(">")[1].split("<")[0]
        
        if len(content)==0:
          break
        else:
          news_content+=content
      content=news_content
      
      ndict["Content"].append(content)

  ekantipur_df = pd.DataFrame(ndict, columns=list(ndict.keys()))
      
  ekantipur_df.to_csv('df.csv')
