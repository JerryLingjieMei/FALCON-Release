import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

method = ['MAC', 'NSCL+GNN', 'FALCON-G'] * 5
percentage = [p  for p in [0, 25, 50, 75, 100] for _ in range(3)]
accuracy = [76.37, 73.38, 73.55, 80.20, 77.16, 73.94, 80.78, 76.93, 74.13, 81.20, 77.77, 74.23, 81.33, 78.50,
    73.88]
data = pd.DataFrame.from_dict({'method': method, 'percentage': percentage, 'accuracy': accuracy})
data.pivot('method','percentage','accuracy')
ax =sns.lineplot(data=data,x='percentage',y='accuracy',hue='method', style='method',
    markers=True, dashes=False)

plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='8') # for legend title
# plt.legend(loc='upper right')
plt.xlabel('accuracy (%)')
plt.xlabel('% of all related concepts in the supplementary sentence')
plt.savefig('output/snippets/fisher.png')
