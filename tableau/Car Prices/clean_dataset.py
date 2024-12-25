import pandas as pd

df = pd.read_csv("./Cars Dataset.csv")

df = df.dropna()



for i in range(0,df.shape[0]): 
    
    if df['Year'].iloc[i] == 20224:
        
        df["Year"].iloc[i] = 2024
    
    
df["Price"] = df["Price"].str.replace(",","")

df["Price"] = df["Price"].astype(float)

df["Price US_Dollar"] = df["Price"]*0.02



df.to_excel("Cars_Dataset_Cleaned.xlsx")