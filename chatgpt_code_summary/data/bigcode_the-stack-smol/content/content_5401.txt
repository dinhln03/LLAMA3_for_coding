from browser import document, alert
import sys
from pprint import pprint

class redirect:
    def write(text, text2):
      document["output"].innerHTML += text2
  
sys.stdout = redirect()     
sys.stderr = redirect()      


d = document["output"]
d.clear()
d.innerHTML = "Hello"

print("Hello again")

def hello(ev):
    alert("Hello !")

document["button1"].bind("click", hello)