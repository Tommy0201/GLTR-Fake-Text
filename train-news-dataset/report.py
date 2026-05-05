import pandas as pd



df1 = pd.read_csv("gpt/gpt_content.csv")
df2 = pd.read_csv("gpt/gpt_titles.csv")

print("shape of df content: ", df1.shape)
print("shape of df titles: ", df2.shape)