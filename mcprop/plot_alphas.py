import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import argparse
import pickle
import numpy as np
import json

figsize=(5,3.1)
legend_loc='lower right'

fig, ax = plt.subplots(figsize=(5, 2))
marker = ['#133248', '#7a972c', '#800000', '#f09d06']
image_alpha_file = 'alpha_plots_data/run-baseline_with-images_weighted-fusion_max-violation_Validation_Alphas_img_alpha-tag-Validation_Alphas.json'
texts_alpha_file = 'alpha_plots_data/run-baseline_with-images_weighted-fusion_max-violation_Validation_Alphas_txt_alpha-tag-Validation_Alphas.json'
values = {}
with open(texts_alpha_file) as sf:
    values['image URL'] = json.load(sf)
with open(image_alpha_file) as rf:
    values['image'] = json.load(rf)

formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel('Iteration')
ax.set_ylabel('Alpha')

for i, (name, p) in enumerate(values.items()):
    it_ndcg = [(k[1], k[2]) for k in p[1:]]
    it, ndcg = zip(*it_ndcg)
    plt.ylim(.3, 1)
    # plt.xticks(xticks)

    ax.plot(it, ndcg, color=marker[i], label=name)
    # plt.plot(sparsity, ir_metric, 's-', color=marker[i], label='ROUGE-L')
plt.grid()
plt.legend(loc=legend_loc, prop={'size': 9})
# plt.show()
# plt.legend()
plt.savefig('alphas.pdf', bbox_inches="tight")