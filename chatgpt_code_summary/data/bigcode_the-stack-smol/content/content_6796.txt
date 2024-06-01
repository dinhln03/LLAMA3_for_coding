################################################
# backend.py is part of COVID.codelongandpros.repl.co
# You should have recieved a copy of the three-clause BSD license. 
# If you did not, it is located at: 
# https://opensource.org/licenses/BSD-3-Clause
# Made by Scott Little, with help from StackOverflow
################################################
import csv
import matplotlib.pyplot as plt
from imageio import imwrite


def get_file():
  url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
  import requests
  r = requests.get(url)

  with open('cases.csv', 'wb') as f:
      f.write(r.content)

  # Retrieve HTTP meta-data
  print(r.status_code)
  print(r.headers['content-type'])
  print(r.encoding)

def get_cases(stat):
  x = []
  y = []
  d = [0]
  dx = [0]
  if len(stat) == 0:
    return 1

  dat = 0

  state = stat
  reader = csv.DictReader(open("cases.csv"))
  for raw in reader:

    if raw['state'] == state:
      dat+=1
      x.append(dat)
      dx.append(dat)
      y.append(raw['cases'])
      d.append(raw['deaths'])
    else:
      continue
  fig, axs = plt.subplots(2,figsize=(12,10))
  fig.suptitle(f"COVID-19 Cases/Deaths in {stat}")
  axs[0].plot(x, y)
  axs[1].plot(dx, d)
  axs[0].set_ylabel('Cases')
  axs[1].set_ylabel("Deaths")
  for axe in axs:

    axe.set_xlabel("Days since 2020-01-21")
  plt.savefig('static/plots/plot.png', bbox_inches='tight', dpi=400)
  return 0

def overwrite():
  import numpy as np
  img = np.zeros([100,100,3],dtype=np.uint8)
  img.fill(255) # or img[:] = 255
  imwrite('static/plots/plot.png', img)