import io
from sdk.db import database
import numpy as np
import matplotlib.pyplot as plt  
from numba import jit
from utils import logger


semester_order = ['FA', 'WI', 'SP', 'SU']

def get_section(x):
    return (
        x.get('semester'),
        x.get('year')
    )

    
def filter_function(sections):
    d = {}
    for x in sections:
        if x not in d.keys():
            d.update({x: 1})
        else:
            d[x] += 1
    return sorted(list(d.items()), key=lambda k: k[0][1] + semester_order.index(k[0][0]) / 10)

def get_plot(sc, cn, isQuarter=True) -> io.BytesIO:
    query = database.get_query()
    database.isQuarter(isQuarter, query)
    database.subject_code(sc, query)
    database.course_number(cn, query)
    q = database.execute(query)
    q = filter_function(map(get_section, q))

    keys = range(0, len(q))
    vals = [x[1] for x in q]

    buffer = io.BytesIO()
    plt.plot(keys, vals)
    plt.xticks(np.arange(len(q)), [f'{x[0][0]} \'{x[0][1]}' for x in q])
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer