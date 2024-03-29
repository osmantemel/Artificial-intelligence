import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

cities = pd.read_csv("california_cities.csv")
cities.head()

latitude ,longitude = cities["latd"] , cities["longd"]

population , area = cities["population_total"] , cities["area_total_km2"]


import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

plt.scatter(longitude, latitude, label=None, c=np.log10(population),
            cmap="viridis", s=area, linewidth=0, alpha=0.5)

plt.axis(aspect="equal")

plt.xlabel("longitude")
plt.ylabel("latitude")

plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area) + ' km$^2$')

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="city areas")
plt.title("area and population")
plt.show()

