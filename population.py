import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st

#)title
st.title(':orange[world population - regression models]')

# reading population dataset
df = pd.read_csv(r'D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\data\world_population_growth.csv')
#df

#) 21. checkbox with text
st.subheader("\n:green[1.knowing the population data🌝]\n")
if (st.checkbox("1. original data")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.1 original dataframe]\n")

    # Sample data
    data = df.head(5)

    # Display the table with highlighting
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))
    st.write(":green[**________________________________________________________________________________________**]")

    #)number of columns
    columns = len(df.columns)
    #)number of rows
    rows = len(df)
    st.markdown("\n#### :red[1.2 number of row and columns]\n")
    st.info(f"{rows} * {columns}")

   
#) to get unique value in country_name column
#df['country_name'].unique()

#) to get the column names
#df.columns

#) breaking the dataframe with respect to year
df_1961 = df[['country_name','country_code','1961']]
df_1962 = df[['country_name','country_code','1962']]
df_1963 = df[['country_name','country_code','1963']]
df_1964 = df[['country_name','country_code','1964']]
df_1965 = df[['country_name','country_code','1965']]
df_1966 = df[['country_name','country_code','1966']]
df_1967 = df[['country_name','country_code','1967']]
df_1968 = df[['country_name','country_code','1968']]
df_1969 = df[['country_name','country_code','1969']]
df_1970 = df[['country_name','country_code','1970']]

df_1971 = df[['country_name','country_code','1971']]
df_1972 = df[['country_name','country_code','1972']]
df_1973 = df[['country_name','country_code','1973']]
df_1974 = df[['country_name','country_code','1974']]
df_1975 = df[['country_name','country_code','1975']]
df_1976 = df[['country_name','country_code','1976']]
df_1977 = df[['country_name','country_code','1977']]
df_1978 = df[['country_name','country_code','1978']]
df_1979 = df[['country_name','country_code','1979']]
df_1980 = df[['country_name','country_code','1980']]

df_1981 = df[['country_name','country_code','1981']]
df_1982 = df[['country_name','country_code','1982']]
df_1983 = df[['country_name','country_code','1983']]
df_1984 = df[['country_name','country_code','1984']]
df_1985 = df[['country_name','country_code','1985']]
df_1986 = df[['country_name','country_code','1986']]
df_1987 = df[['country_name','country_code','1987']]
df_1988 = df[['country_name','country_code','1988']]
df_1989 = df[['country_name','country_code','1989']]
df_1990 = df[['country_name','country_code','1990']]

df_1991 = df[['country_name','country_code','1991']]
df_1992 = df[['country_name','country_code','1992']]
df_1993 = df[['country_name','country_code','1993']]
df_1994 = df[['country_name','country_code','1994']]
df_1995 = df[['country_name','country_code','1995']]
df_1996 = df[['country_name','country_code','1996']]
df_1997 = df[['country_name','country_code','1997']]
df_1998 = df[['country_name','country_code','1998']]
df_1999 = df[['country_name','country_code','1999']]
df_2000 = df[['country_name','country_code','2000']]

df_2001 = df[['country_name','country_code','2001']]
df_2002 = df[['country_name','country_code','2002']]
df_2003 = df[['country_name','country_code','2003']]
df_2004 = df[['country_name','country_code','2004']]
df_2005 = df[['country_name','country_code','2005']]
df_2006 = df[['country_name','country_code','2006']]
df_2007 = df[['country_name','country_code','2007']]
df_2008 = df[['country_name','country_code','2008']]
df_2009 = df[['country_name','country_code','2009']]
df_2010 = df[['country_name','country_code','2010']]

df_2011 = df[['country_name','country_code','2011']]
df_2012 = df[['country_name','country_code','2012']]
df_2013 = df[['country_name','country_code','2013']]
df_2014 = df[['country_name','country_code','2014']]
df_2015 = df[['country_name','country_code','2015']]
df_2016 = df[['country_name','country_code','2016']]
df_2017 = df[['country_name','country_code','2017']]
df_2018 = df[['country_name','country_code','2018']]
df_2019 = df[['country_name','country_code','2019']]
df_2020 = df[['country_name','country_code','2020']]
df_2021 = df[['country_name','country_code','2021']]
df_2022 = df[['country_name','country_code','2022']]

#)1961
df_1961 = pd.DataFrame()
df_1961 = df[['country_name','country_code','1961']]
df_1961['year'] = 1961
#df_1961

#) making uniform name to all the new dataframe
df_1961.rename(columns = {'1961':'population'}, inplace = True)
#df_1961

#)1962
df_1962 = pd.DataFrame()
df_1962 = df[['country_name','country_code','1962']]
df_1962['year'] = 1962
#df_1962

#) making uniform name to all the new dataframe
df_1962.rename(columns = {'1962':'population'}, inplace = True)
#df_1962

#)1963
df_1963 = pd.DataFrame()
df_1963 = df[['country_name','country_code','1963']]
df_1963['year'] = 1963
#df_1963

#) making uniform name to all the new dataframe
df_1963.rename(columns = {'1963':'population'}, inplace = True)
#df_1963

#)1964
df_1964 = pd.DataFrame()
df_1964 = df[['country_name','country_code','1964']]
df_1964['year'] = 1964
#df_1964

#) making uniform name to all the new dataframe
df_1964.rename(columns = {'1964':'population'}, inplace = True)
#df_1964

#)1965
df_1965 = pd.DataFrame()
df_1965 = df[['country_name','country_code','1965']]
df_1965['year'] = 1965
#df_1965

#) making uniform name to all the new dataframe
df_1965.rename(columns = {'1965':'population'}, inplace = True)
#df_1965

#)1966
df_1966 = pd.DataFrame()
df_1966 = df[['country_name','country_code','1966']]
df_1966['year'] = 1966
#df_1966

#)1966
df_1966 = pd.DataFrame()
df_1966 = df[['country_name','country_code','1966']]
df_1966['year'] = 1966
#df_1966

#) making uniform name to all the new dataframe
df_1966.rename(columns = {'1966':'population'}, inplace = True)
#df_1966

#)1967
df_1967 = pd.DataFrame()
df_1967 = df[['country_name','country_code','1967']]
df_1967['year'] = 1967
#df_1967

#) making uniform name to all the new dataframe
df_1967.rename(columns = {'1967':'population'}, inplace = True)
#df_1967

#)1968
df_1968 = pd.DataFrame()
df_1968 = df[['country_name','country_code','1968']]
df_1968['year'] = 1968
#df_1968

#) making uniform name to all the new dataframe
df_1968.rename(columns = {'1968':'population'}, inplace = True)
#df_1968

#)1969
df_1969 = pd.DataFrame()
df_1969 = df[['country_name','country_code','1969']]
df_1969['year'] = 1969
#df_1969

#) making uniform name to all the new dataframe
df_1969.rename(columns = {'1969':'population'}, inplace = True)
#df_1969

#)1970
df_1970 = pd.DataFrame()
df_1970 = df[['country_name','country_code','1970']]
df_1970['year'] =1970
#df_1970

#) making uniform name to all the new dataframe
df_1970.rename(columns = {'1970':'population'}, inplace = True)
#df_1970

#)1971
df_1971 = pd.DataFrame()
df_1971 = df[['country_name','country_code','1971']]
df_1971['year'] = 1971
#df_1971

df_1971.rename(columns = {'1971':'population'}, inplace = True)
#df_1971

#)1972
df_1972 = pd.DataFrame()
df_1972 = df[['country_name','country_code','1972']]
df_1972['year'] = 1972
#df_1972

df_1972.rename(columns = {'1972':'population'}, inplace = True)
#df_1972

#)1973
df_1973 = pd.DataFrame()
df_1973 = df[['country_name','country_code','1973']]
df_1973['year'] = 1973
#df_1973

df_1973.rename(columns = {'1973':'population'}, inplace = True)
#df_1973

#)1964
df_1974 = pd.DataFrame()
df_1974 = df[['country_name','country_code','1974']]
df_1974['year'] = 1974
#df_1974

df_1974.rename(columns = {'1974':'population'}, inplace = True)
#df_1974

#)1965
df_1975 = pd.DataFrame()
df_1975 = df[['country_name','country_code','1975']]
df_1975['year'] = 1975
#df_1975

df_1975.rename(columns = {'1975':'population'}, inplace = True)
#df_1975

#)1966
df_1976 = pd.DataFrame()
df_1976 = df[['country_name','country_code','1976']]
df_1976['year'] = 1976
#df_1976

df_1976.rename(columns = {'1976':'population'}, inplace = True)
#df_1976

#)1977
df_1977 = pd.DataFrame()
df_1977 = df[['country_name','country_code','1977']]
df_1977['year'] = 1977
#df_1977

df_1977.rename(columns = {'1977':'population'}, inplace = True)
#df_1977

#)1978
df_1978 = pd.DataFrame()
df_1978 = df[['country_name','country_code','1978']]
df_1978['year'] = 1978
#df_1978

df_1978.rename(columns = {'1978':'population'}, inplace = True)
#df_1978

#)1979
df_1979 = pd.DataFrame()
df_1979 = df[['country_name','country_code','1979']]
df_1979['year'] = 1979
#df_1979

df_1979.rename(columns = {'1979':'population'}, inplace = True)
#df_1979

#)1980
df_1980 = pd.DataFrame()
df_1980 = df[['country_name','country_code','1980']]
df_1980['year'] =1980
#df_1980

df_1980.rename(columns = {'1980':'population'}, inplace = True)
#df_1980

#)1981
df_1981 = pd.DataFrame()
df_1981 = df[['country_name','country_code','1981']]
df_1981['year'] = 1981
#df_1981

df_1981.rename(columns = {'1981':'population'}, inplace = True)
#df_1981

#)1982
df_1982 = pd.DataFrame()
df_1982 = df[['country_name','country_code','1982']]
df_1982['year'] = 1982
#df_1982

df_1982.rename(columns = {'1982':'population'}, inplace = True)
#df_1982

#)1983
df_1983 = pd.DataFrame()
df_1983 = df[['country_name','country_code','1983']]
df_1983['year'] = 1983
#df_1983

df_1983.rename(columns = {'1983':'population'}, inplace = True)
#df_1983

#)1984
df_1984 = pd.DataFrame()
df_1984 = df[['country_name','country_code','1984']]
df_1984['year'] = 1984
#df_1984

df_1984.rename(columns = {'1984':'population'}, inplace = True)
#df_1984

#)1985
df_1985 = pd.DataFrame()
df_1985 = df[['country_name','country_code','1985']]
df_1985['year'] = 1985
#df_1985

df_1985.rename(columns = {'1985':'population'}, inplace = True)
#df_1985

#)1986
df_1986 = pd.DataFrame()
df_1986 = df[['country_name','country_code','1986']]
df_1986['year'] = 1986
#df_1986

df_1986.rename(columns = {'1986':'population'}, inplace = True)
#df_1986

#)1987
df_1987 = pd.DataFrame()
df_1987 = df[['country_name','country_code','1987']]
df_1987['year'] = 1987
#df_1987

df_1987.rename(columns = {'1987':'population'}, inplace = True)
#df_1987

#)1988
df_1988 = pd.DataFrame()
df_1988 = df[['country_name','country_code','1988']]
df_1988['year'] = 1988
#df_1988

df_1988.rename(columns = {'1988':'population'}, inplace = True)
#df_1988

#)1989
df_1989 = pd.DataFrame()
df_1989 = df[['country_name','country_code','1989']]
df_1989['year'] = 1989
#df_1989

df_1989.rename(columns = {'1989':'population'}, inplace = True)
#df_1989

#)1980
df_1990 = pd.DataFrame()
df_1990 = df[['country_name','country_code','1990']]
df_1990['year'] =1990
#df_1990

df_1990.rename(columns = {'1990':'population'}, inplace = True)
#df_1990

#)1991
df_1991 = pd.DataFrame()
df_1991 = df[['country_name','country_code','1991']]
df_1991['year'] = 1991
#df_1991

df_1991.rename(columns = {'1991':'population'}, inplace = True)
#df_1991

#)1992
df_1992 = pd.DataFrame()
df_1992 = df[['country_name','country_code','1992']]
df_1992['year'] = 1992
#df_1992

df_1992.rename(columns = {'1992':'population'}, inplace = True)
#df_1992

#)1993
df_1993 = pd.DataFrame()
df_1993 = df[['country_name','country_code','1993']]
df_1993['year'] = 1993
#df_1993

df_1993.rename(columns = {'1993':'population'}, inplace = True)
#df_1993

#)1994
df_1994 = pd.DataFrame()
df_1994 = df[['country_name','country_code','1994']]
df_1994['year'] = 1994
#df_1994

df_1994.rename(columns = {'1994':'population'}, inplace = True)
#df_1994

#)1995
df_1995 = pd.DataFrame()
df_1995 = df[['country_name','country_code','1995']]
df_1995['year'] = 1995
#df_1995

df_1995.rename(columns = {'1995':'population'}, inplace = True)
#df_1995

#)1996
df_1996 = pd.DataFrame()
df_1996 = df[['country_name','country_code','1996']]
df_1996['year'] = 1996
#df_1996

df_1996.rename(columns = {'1996':'population'}, inplace = True)
#df_1996

#)1997
df_1997 = pd.DataFrame()
df_1997 = df[['country_name','country_code','1997']]
df_1997['year'] = 1997
#df_1997

df_1997.rename(columns = {'1997':'population'}, inplace = True)
#df_1997

#)1988
df_1998 = pd.DataFrame()
df_1998 = df[['country_name','country_code','1998']]
df_1998['year'] = 1998
#df_1998

df_1998.rename(columns = {'1998':'population'}, inplace = True)
#df_1998

#)1999
df_1999 = pd.DataFrame()
df_1999 = df[['country_name','country_code','1999']]
df_1999['year'] = 1999
#df_1999

df_1999.rename(columns = {'1999':'population'}, inplace = True)
#df_1999

#)2000
df_2000 = pd.DataFrame()
df_2000 = df[['country_name','country_code','2000']]
df_2000['year'] =2000
#df_2000

df_2000.rename(columns = {'2000':'population'}, inplace = True)
#df_2000

#)2001
df_2001 = pd.DataFrame()
df_2001 = df[['country_name','country_code','2001']]
df_2001['year'] = 2001
#df_2001

df_2001.rename(columns = {'2001':'population'}, inplace = True)
#df_2001

#)2002
df_2002 = pd.DataFrame()
df_2002 = df[['country_name','country_code','2002']]
df_2002['year'] = 2002
#df_2002

df_2002.rename(columns = {'2002':'population'}, inplace = True)
#df_2002

#)2003
df_2003 = pd.DataFrame()
df_2003 = df[['country_name','country_code','2003']]
df_2003['year'] = 2003
#df_2003

df_2003.rename(columns = {'2003':'population'}, inplace = True)
#df_2003

#)2004
df_2004 = pd.DataFrame()
df_2004 = df[['country_name','country_code','2004']]
df_2004['year'] = 2004
#df_2004

df_2004.rename(columns = {'2004':'population'}, inplace = True)
#df_2004

#)2005
df_2005 = pd.DataFrame()
df_2005 = df[['country_name','country_code','2005']]
df_2005['year'] = 2005
#df_2005

df_2005.rename(columns = {'2005':'population'}, inplace = True)
#df_2005

#)2006
df_2006 = pd.DataFrame()
df_2006 = df[['country_name','country_code','2006']]
df_2006['year'] = 2006
#df_2006

df_2006.rename(columns = {'2006':'population'}, inplace = True)
#df_2006

#)2007
df_2007 = pd.DataFrame()
df_2007 = df[['country_name','country_code','2007']]
df_2007['year'] = 2007
#df_2007

df_2007.rename(columns = {'2007':'population'}, inplace = True)
#df_2007

#)2008
df_2008 = pd.DataFrame()
df_2008 = df[['country_name','country_code','2008']]
df_2008['year'] = 2008
#df_2008

df_2008.rename(columns = {'2008':'population'}, inplace = True)
#df_2008

#)2009
df_2009 = pd.DataFrame()
df_2009 = df[['country_name','country_code','2009']]
df_2009['year'] = 2009
#df_2009

df_2009.rename(columns = {'2009':'population'}, inplace = True)
#df_2009

#)2010
df_2010 = pd.DataFrame()
df_2010 = df[['country_name','country_code','2010']]
df_2010['year'] =2010
#df_2010

df_2010.rename(columns = {'2010':'population'}, inplace = True)
#df_2010

#)2011
df_2011 = pd.DataFrame()
df_2011 = df[['country_name','country_code','2011']]
df_2011['year'] = 2011
#df_2011

df_2011.rename(columns = {'2011':'population'}, inplace = True)
#df_2011

#)2012
df_2012 = pd.DataFrame()
df_2012 = df[['country_name','country_code','2012']]
df_2012['year'] = 2012
#df_2002

df_2012.rename(columns = {'2012':'population'}, inplace = True)
#df_2012

#)2013
df_2013 = pd.DataFrame()
df_2013 = df[['country_name','country_code','2013']]
df_2013['year'] = 2013
#df_2013

df_2013.rename(columns = {'2013':'population'}, inplace = True)
#df_2013

#)2014
df_2014 = pd.DataFrame()
df_2014 = df[['country_name','country_code','2014']]
df_2014['year'] = 2014
#df_2014

df_2014.rename(columns = {'2014':'population'}, inplace = True)
#df_2014

#)2015
df_2015 = pd.DataFrame()
df_2015 = df[['country_name','country_code','2015']]
df_2015['year'] = 2015
#df_2015

df_2015.rename(columns = {'2015':'population'}, inplace = True)
#df_2015

#)2016
df_2016 = pd.DataFrame()
df_2016 = df[['country_name','country_code','2016']]
df_2016['year'] = 2016
#df_2016

df_2016.rename(columns = {'2016':'population'}, inplace = True)
#df_2016

#)2017
df_2017 = pd.DataFrame()
df_2017 = df[['country_name','country_code','2017']]
df_2017['year'] = 2017
#df_2017

df_2017.rename(columns = {'2017':'population'}, inplace = True)
#df_2017

#)2018
df_2018 = pd.DataFrame()
df_2018 = df[['country_name','country_code','2018']]
df_2018['year'] = 2018
#df_2018

df_2018.rename(columns = {'2018':'population'}, inplace = True)
#df_2018

#)2019
df_2019 = pd.DataFrame()
df_2019 = df[['country_name','country_code','2019']]
df_2019['year'] = 2019
#df_2019

df_2019.rename(columns = {'2019':'population'}, inplace = True)
#df_2019

#)2020
df_2020 = pd.DataFrame()
df_2020 = df[['country_name','country_code','2020']]
df_2020['year'] =2020
#df_2020

df_2020.rename(columns = {'2020':'population'}, inplace = True)
#df_2020

#)2021
df_2021 = pd.DataFrame()
df_2021 = df[['country_name','country_code','2021']]
df_2021['year'] = 2021
#df_2021

df_2021.rename(columns = {'2021':'population'}, inplace = True)
#df_2021

#)2022
df_2022 = pd.DataFrame()
df_2022 = df[['country_name','country_code','2022']]
df_2022['year'] = 2022
#df_2022

df_2022.rename(columns = {'2022':'population'}, inplace = True)
#df_2022

#) concat all the broken frame 
result_df = pd.concat([df_1961, df_1962, df_1964,df_1965, df_1966,df_1967, df_1968,df_1969, df_1970,
                       df_1971,df_1972,df_1973, df_1974,df_1975, df_1976,df_1977, df_1978,df_1979, df_1980,
                       df_1981, df_1982,df_1983, df_1984,df_1985, df_1986,df_1987, df_1988,df_1989, df_1990,
                       df_1991, df_1992,df_1993, df_1994,df_1995, df_1996,df_1997, df_1998,df_1999, df_2000,
                       df_2001, df_2002,df_2003, df_2004,df_2005, df_2006,df_2007, df_2008,df_2009, df_2010,
                       df_2011, df_2012,df_2013, df_2014,df_2015, df_2016,df_2017, df_2018,df_2019, df_2020,
                       df_2021, df_2022], axis=0, ignore_index=True)
#result_df

if (st.checkbox("2. data post processing")):
    #)showing original dataframe
    st.markdown("\n#### :blue[2.1 dataframe post processing]\n")

    # Sample data
    data = result_df.head(5)

    # Display the table with highlighting
    st.dataframe(data.style.applymap(lambda x: 'color:blue'))
    st.write(":blue[**________________________________________________________________________________________**]")

    #)number of columns
    columns = len(result_df.columns)
    #)number of rows
    rows = len(result_df)
    st.markdown("\n#### :red[2.2 number of row and columns]\n")
    st.success(f"{rows} * {columns}")

#) converting df to list
list_year = result_df['year'].tolist()
list_population = result_df['population'].tolist()

np_year = np.array(list_year)
np_population = np.array(list_population)

#) converting list to np
if (st.checkbox("2. scatterplot ")):
    #)showing original dataframe
    st.markdown("\n#### :blue[3. scatterplot]\n")
    fig,ax = plt.subplots(figsize=(15,8))
    ax.scatter(np_year,np_population,color = 'yellow')
    st.pyplot(fig)

#) to check unique value in year column
#result_df['year'].unique()

#) checking the null value
result_df.isna().sum()

#) filling na 
temp_df = result_df.ffill()
temp_df = result_df.bfill()
#temp_df

#)to check the null value
temp_df.isna().sum()

#) to get unique value in in country_code column
result_df['country_code'].unique()

#) dropping country_code column
temp_df.drop(['country_code'],axis=1,inplace=True)

#)applying dummies
temp_df = pd.get_dummies(temp_df,['country_name'])

X = temp_df.drop(['population'],axis=1)
y = temp_df['population']

st.subheader(':orange[2.regression models🌞]')
selectBox=st.selectbox(":violet[**regression models:**] ", ['LinearRegression',
                                                            'Ridge',
                                                            'Lasso'])

if selectBox == 'LinearRegression':
    st.markdown("\n#### :blue[2.1 LinearRegression]\n")
    from sklearn.linear_model import LinearRegression
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = LinearRegression()
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    st.success("********Train data(mse)*********")
    st.write(mean_squared_error(y_train,train_pred))
    
    st.success("********Test data(mse)*********")
    st.write(mean_squared_error(y_test,test_pred))
    
    st.info("actual training vs training prediction")
    train_df = pd.DataFrame()
    train_df['train_actual']= y_train
    train_df['train_pred'] = train_pred
    data1 = train_df.head(4)
    st.dataframe(data1.style.applymap(lambda x: 'color:red'))
    
    st.info("actual training vs training prediction")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    data2 = test_df.head(4)
    st.dataframe(data2.style.applymap(lambda x: 'color:green'))

#) Ridge regression model
elif selectBox == 'Ridge':
    st.markdown("\n#### :violet[2.2 Ridge]\n")
    from sklearn.linear_model import Ridge
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = Ridge()
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    st.info("********Train data(mse)*********")
    st.write(mean_squared_error(y_train,train_pred))
    
    st.info("********Test data(mse)*********")
    st.write(mean_squared_error(y_test,test_pred))
    
    st.error("actual training vs training prediction")
    train_df = pd.DataFrame()
    train_df['train_actual']= y_train
    train_df['train_pred'] = train_pred
    data1 = train_df.head(4)
    st.dataframe(data1.style.applymap(lambda x: 'color:green'))
    
    st.error("actual training vs training prediction")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    data2 = test_df.head(4)
    st.dataframe(data2.style.applymap(lambda x: 'color:red'))

#) Ridge regression model
elif selectBox == 'Lasso':
    st.markdown("\n#### :purple[2.3 Lasso]\n")
    from sklearn.linear_model import Lasso
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = Lasso()
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    st.write(":orange[********Train data(mse)*********]")
    st.info(mean_squared_error(y_train,train_pred))
    
    st.write(":orange[********Test data(mse)*********]")
    st.info(mean_squared_error(y_test,test_pred))
    
    st.write(":green[actual training vs training prediction]")
    train_df = pd.DataFrame()
    train_df['train_actual']= y_train
    train_df['train_pred'] = train_pred
    data1 = train_df.head(4)
    st.dataframe(data1.style.applymap(lambda x: 'color:blue'))
    
    st.write(":violet[actual training vs training prediction]")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    data2 = test_df.head(4)
    st.dataframe(data2.style.applymap(lambda x: 'color:orange'))