# coding:utf-8
from __future__ import unicode_literals

from bs4 import BeautifulSoup
import requests
import re

def get_total(target_url):
    req=requests.get(url=target_url)
    req.encoding='utf-8'
    htlm=req.text
    div_bf = BeautifulSoup(htlm, 'lxml')
    div = div_bf.find_all('div', attrs={'class':'content1'})
    return div

def get_table(div_raw):
    table_list = []
    for i in div_raw:
        table_list.append(i.find_all('tr'))
    text_list = []
    for i in table_list:
        for j in i:
            text = j.get_text()
            text = re.sub('\s','',text)
            sub_list = text.split('·')
            if '' in sub_list:
                sub_list.remove('')
            if sub_list:
                text_list.append(sub_list)
    return text_list

def get_dict(text_raw):
    content_dict = {}
    for i in text_raw:
        for j in i:
            sub_list = j.split('：')
            if len(sub_list) >  1:
                content_dict.update({sub_list[0]:sub_list[1]})
            else:
                content_dict.update({sub_list[0]:''})
    return content_dict

def find_item(find_list,target):
    info=[]
    company_dict = get_dict(get_table(get_total(target)))
    for key, value in company_dict.items():
        for i in find_list:
            if re.search(i,key):
                info.append(value.encode('utf-8'))
    return info

