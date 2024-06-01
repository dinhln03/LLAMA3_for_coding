# Data from https://www.kaggle.com/crawford/80-cereals/version/2
import pandas, matplotlib
data = pandas.read_csv('http://www.compassmentis.com/wp-content/uploads/2019/04/cereal.csv')
data = data.set_index('name')
data = data.calories.sort_values()[-10:]
ax = data.plot(kind='barh')
ax.set_xlabel('Calories per serving')
ax.set_ylabel('Cereal')
ax.set_title('Top 10 cereals by calories')
matplotlib.pyplot.subplots_adjust(left=0.25)
matplotlib.pyplot.show()
