import streamlit as st

import plotly.express as px
import string
import pandas as pd






st.title("Smartphone Sales Dashboard")


df = pd.read_csv("Sales.csv")



df = df.dropna()


df["Storage"] = df["Storage"].str.replace('GB','')
df["Memory"] = df["Memory"].str.replace('GB','')


df["Storage"] = pd.to_numeric(df["Storage"],errors = "coerce")
df["Memory"] =  pd.to_numeric(df["Memory"],errors = "coerce")

df = df.dropna()



def filter_out(df,column_name,k):
    
    """
    
        applying filters based on user input
    
    
    """
    
    
    container = st.container()
    
    all_ = st.checkbox("Select all",key = k)
 
    if not all_:
        
        
        all_options = df[column_name].unique()
    
        selected_options = container.multiselect(column_name, all_options,label_visibility="collapsed")
    
        df = df.loc[(df[column_name].isin(selected_options))]
    


    
    return df





def group_df(df):

    """
        groups dataframe 
    
    """

    pass


    return df




def show(agg_df,y_column,my_title):
    
    agg_df = agg_df.reset_index()
    
    agg_df['Name'] = agg_df['Models'] + ' ' + agg_df['Colors']

    b = px.bar(data_frame = agg_df, 
               x = "Name",
               y = y_column,
               color = "Brands",
               title = my_title)
    
    st.plotly_chart(b)
    
    
    return






st.header("Filter")
c1 = st.container(border = True)


with c1: 
    
    # Filter by brands 
    st.subheader("Filter by Brands")
    
    df = filter_out(df,"Brands",1)
    
    
    
    st.subheader("Filter by Models")
    
    df = filter_out(df,"Models",2)
    
    
    st.subheader("Filter by Color")
    
    df = filter_out(df,"Colors",3)


for i in range(5):
    st.write("\n")


st.header("🧾 Filtered Table")
c2 = st.container(border = True)
c2.dataframe(df)







# group by mean Selling Price 
sub_df = df[["Brands","Models","Colors","Selling Price"]]
agg_df = sub_df.groupby(by = ["Brands","Models","Colors"]).mean()


# group by mean Original Price 
sub_df = df[["Brands","Models","Colors","Original Price"]]
agg_df2 = sub_df.groupby(by = ["Brands","Models","Colors"]).mean()


#group by mean rating and amount of raitings 
sub_df = df[["Brands","Models","Colors","Rating"]]
agg_df3 = sub_df.groupby(by = ["Brands","Models","Colors"]).mean()


#group by mean rating and amount of raitings 
sub_df = df[["Brands","Models","Colors","Rating"]]
agg_df4 = sub_df.groupby(by = ["Brands","Models","Colors"]).size()
agg_df4 = agg_df4.to_frame()
agg_df4 = agg_df4.rename(columns = {0:"count"})




# group by color
sub_df = df[["Brands","Models","Colors"]]
agg_df6 = sub_df.groupby(by = ["Brands","Models"]).size()




for i in range(5):
    st.write("\n")



st.header("📊 Diagrams")


col1,col2 = st.columns(2,border=True)


with col1:
    
    show(agg_df,"Selling Price",'Average Selling Price')



with col2: 

    show(agg_df2,"Original Price","Average Original Price")






col3,col4 = st.columns(2,border=True)

with col3: 
    
    show(agg_df3,"Rating","Average Rating")


    
with col4:
    
    show(agg_df4,"count","Amount of Ratings")


# px.bar(agg_df,)
