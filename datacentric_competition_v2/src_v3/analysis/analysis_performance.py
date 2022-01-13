import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('max_column', None)
sub_1208 = pd.read_csv("../performance_df_submission_1208.tsv", sep='\t')
sub_2708 = pd.read_csv("../performance_df_submission_2708_v2.tsv", sep='\t')

print(sub_1208.mean().mean())
print(sub_2708.mean().mean())