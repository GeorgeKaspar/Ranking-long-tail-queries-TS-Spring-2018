import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool


def parallelize(pool, data, func, partitions):
    data_split = np.array_split(data, partitions)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data
