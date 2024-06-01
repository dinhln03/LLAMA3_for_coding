from IMDB_task4 import scrape_movie_details
from pprint import pprint
import os,requests,json,time,random
from IMDB_task1 import scrape_top_list
# task13
# this task for the make a json file ini our directory
def save_data():
	movies_data = scrape_top_list()
	for one_movie in movies_data :
		id_movie = (one_movie['urls'][-10:-1])
		exists = os.path.exists("screpingdata/" + str(id_movie) + ".json")
		cwd = os.getcwd()
		if exists:
			with open(cwd+"/screpingdata/" + str(id_movie) + ".json","r+") as file :
				data = file.read()
				load_data = json.loads(data)
				return (load_data)
		else:
			for one_movie in movies_data :
				id_movie = (one_movie['urls'][-10:-1])
				# task_no. 9
				sleep_time = random.randint(1,3)
				time.sleep(sleep_time)
				url = (one_movie["urls"])
				screpe_movie_data = scrape_movie_details(url)
				with open("screpingdata/" + str(id_movie) + ".json","w") as file :
					data = json.dumps(screpe_movie_data,indent=4, sort_keys = True)
					write_data = file.write(data)
					return (write_data)
pprint (save_data())
