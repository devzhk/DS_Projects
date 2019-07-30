import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

result1 = pd.read_csv('Results/paintym.csv')
palette = sns.color_palette("mako_r", 2)
sns.set_style("whitegrid")
sns.lineplot(x="dimension", y="score",hue="method",style='method',markers=True,dashes=False,data=result1,palette=palette)
plt.show()
print("hello")