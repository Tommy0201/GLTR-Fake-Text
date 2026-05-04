import pandas as pd

df = pd.read_csv("articles_content_10k.csv")

mid= len(df) // 2

df_even = df.iloc[::2]
df_odd= df.iloc[1::2]

df_even.to_csv("articles_content_even.csv", index=False)
