import unittest,os
from src.tasks.scrape_reddit.tiktok import dwn_tiktok
from src.tasks.generate_video.task import generate_tiktok
from src.tasks.upload_video.task import upload_video

class TestTiktok(unittest.TestCase):

    def setUp(self):
        pass



    def test_tiktok(self):
        context = {
            'page':{
                'Nombre':"Pagina que hace compilaciones perronas de tiktok",
                "thumbnail": False,
                'description':['Y a ti te ha pasado eso? \nIngresa mi codigo para que ganes dinero!!\nKwai 848290921'],
                'tags':['Amor','Meme','Chistes','Divertido','Reddit'],
                "playlist":"Compilaciones TikTok"
            },
            'video_path':os.getcwd()+"\\"+r"test\test_videos\caption.mp4",
            'thumbnail_path':os.getcwd()+"\\"+r"data\thumbnails\8ccd23f7-7292-41d7-a743-b2c9f2b7fd36.png"}
        #dwn_tiktok(context)
        #generate_tiktok(context)
        upload_video(context)