#importing pandas package
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
# making data frame from csv file
dataset = pd.read_csv('curve.csv', sep=',')

df = pd.DataFrame(dataset.sort_values('Split')[['Split','Score']]) 
df.reset_index(drop=True, inplace=True)



print(df)

#exit()

sns.set_theme(style="darkgrid")

plt.figure(figsize=(65,20))

xlabel = (['0.005','0.010','0.015','0.020','0.040','0.060','0.080','0.1','0.15','0.20','0.25','0.30','0.35','0.40','0.45','0.50','0.55','0.60'
                    ,'0.65','0.70','0.75','0.80','0.85','0.90','0.95','0.100'])

ax = sns.lineplot(x=xlabel, y="Score",
             data=df,color='green',marker=".")

ax.set_xticklabels(labels=xlabel,rotation=45)

ax.set(xlabel='Split')

plt.show()