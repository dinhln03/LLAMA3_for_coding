import urllib.request
import os
import random
import socket

def url_open(url):
   #代理
   iplist=['60.251.63.159:8080','118.180.15.152:8102','119.6.136.122:80','183.61.71.112:8888']
   proxys= random.choice(iplist)
   print (proxys)
   proxy_support = urllib.request.ProxyHandler({'http': proxys})
   opener = urllib.request.build_opener(proxy_support)
   opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36')]
   urllib.request.install_opener(opener)

   #头文件 
   head={}
   head['User-Agent']='Mozilla/5.0 (Windows NT 6.3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'

   req=urllib.request.Request(url,headers=head)
   res = urllib.request.urlopen(req)
   html = res.read()
   print(url)
   return html

def get_page(url):
   html=url_open(url).decode('utf-8')

   a = html.find('current-comment-page')+23
   b = html.find(']',a)
   
   return html[a:b]


def find_imgs(url):
    html = url_open(url).decode('utf-8')
    img_addrs = []

    a = html.find('img src=')
    
    while a != -1:
        b = html.find('.jpg',a,a+255) 
        if b != -1:
            img_addrs.append(html[a+9:b+4])
        else:
            b=a+9
            
        a=html.find('img src=',b)
        
    #for each in img_addrs:
    #    print(each)
    #return img_addrs


def save_imgs(folder,img_addrs):
    socket.setdefaulttimeout(3)
    
    for each in img_addrs:
        try:
            filename = each.split('/')[-1]
            with open(filename,'wb') as f:
                img = url_open(each)
                f.write(img)
        except Exception:
            continue

def download_mm(folder = 'ooxx',pages=10):
    os.mkdir(folder)
    os.chdir(folder)
    
    url='http://jandan.net/ooxx/'
    
    #拿到所在页面
    page_num= int(get_page(url))

    for i in range(pages):
        page_num = page_num - i
        page_url = url+'page-'+str(page_num)+'#comments'

        #查询页面中的图片
        img_addrs = find_imgs(page_url)

        #保存图片
        save_imgs(folder,img_addrs)

if __name__ == '__main__':
    download_mm()



