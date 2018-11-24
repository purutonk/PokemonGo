#importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#fetching data into dataset
dataset=pd.read_csv('300k.csv') 
ylong= dataset.iloc[:,2:3].values
ylat= dataset.iloc[:,1:2].values


''' FIRST PROBLEM SET '''              
                  
#importing library for ploting on map                  
from mpl_toolkits.basemap import Basemap
# Defining the projection, scale, the corners of the map, and the resolution.
m = Basemap(projection='merc',llcrnrlat=-55,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

# Coloring the continents
m.fillcontinents(color='grey',lake_color='grey')

#drawing boundary for countries
m.drawcountries(linewidth=0.1)

# fill in the oceans
m.drawmapboundary(fill_color='black')

#scattering the pokemons
m.scatter(ylong,ylat,latlon=True,marker='.',color='blue',zorder=1)
plt.title("scattering pokemons")
plt.show()

''' SECOND PROBLEM SET '''

city=dataset.iloc[:,21]    #fetching city coloumn
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
city_encoded= le.fit_transform(city)

#calculating city frequency
import collections
frequency= dict (collections.Counter(city_encoded))
#print frequency

city_names=dataset.iloc[:,21] 

frequency_value=list(frequency.values())
frequency_key=list(frequency.keys())
#print frequency_key[frequency_value.index(max(frequency_value))]   #encoded value of city having maximum number of frequency


city_frequency = collections.Counter(city_names)
print "The city having maximum number of pokemon and its frequency is:",city_frequency.most_common(1)
print "Cities having highest number of pokemon count are:",city_frequency.most_common(10)

#plotting bar graph
xlocations = np.array(range(len(frequency_value)))+0.5
width = 0.5
plt.bar(xlocations, frequency_value, width=width)
plt.yticks(range(0, 8))
plt.xticks(xlocations+ width/2, city_names)
plt.xlim(0, xlocations[-1]+width*2)
plt.title("number of pokemons per city")
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.show()


q=zip(city_names, city_encoded)

import seaborn as sns

sns.axes_style('white')
sns.set_style('white')

colors = ['pink' if _y >=0 else 'red' for _y in frequency_value]
ax = sns.barplot(city_names, frequency_value, palette=colors)

for n, (label, _y) in enumerate(zip(city_names, frequency_value)):
    ax.annotate(
        s='{:.1f}'.format(abs(_y)),
        xy=(n, _y),
        ha='center',va='center',
        xytext=(0,10),
        textcoords='offset points',
        color='white',
        weight='bold'
    )

    ax.annotate(
        s=label,
        xy=(n, 0),
        ha='center',va='center',
        xytext=(0,10),
        textcoords='offset points',
    )  
# axes formatting
ax.set_yticks([])
ax.set_xticks([])
sns.despine(ax=ax, bottom=True, left=True)

''' THIRD PROBLEM SET '''

df_latitude=pd.DataFrame(dataset['latitude'][dataset['continent']=='Asia'])
df_longitude=pd.DataFrame(dataset['longitude'][dataset['continent']=='Asia'])
df_lat_long=df_latitude.join(df_longitude)

#dataset=dataset.drop("Asiaa",axis=1)
df_continent=pd.DataFrame(dataset['continent'])
'''dataset['Asiaa']=dataset['continent']=='Asia'
dfa=pd.DataFrame(dataset['Asiaa'])'''

df_id_asia=pd.DataFrame(dataset['pokemonId'][dataset['continent']=='Asia'])

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(df_id_asia,df_lat_long)


#y_pred=regressor.predict(input("enter id of pokemon in asia  and get its location"))
y_pred=regressor.predict(99)
print "Your pokemon is at(classification by continents)",y_pred


'''FOURTH PROBLEM SET'''
#1st generation pokemon names(1-151)
Names=['Bulbasaur','Ivysaur','Venusaur','Charmander','Charmeleon','Charizard','Squirtle','Wartortle','Blastoise','Caterpie','Metapod','Butterfree','Weedle','Kakuna','Beedrill','Pidgey','Pidgeotto','Pidgeot','Rattata','Raticate','Spearow','Fearow','Ekans','Arbok','Pikachu','Raichu','Sandshrew','Sandslash','Nidoran','Nidorina','Nidoqueen','Nidoran♂','Nidorino','Nidoking','Clefairy','Clefable','Vulpix','Ninetales','Jigglypuff','Wigglytuff','Zubat','Golbat','Oddish','Gloom','Vileplume','Paras','Parasect','Venonat','Venomoth','Diglett','Dugtrio','Meowth','Persian','Psyduck','Golduck','Mankey','Primeape','Growlithe','Arcanine','Poliwag','Poliwhirl','Poliwrath','Abra','Kadabra','Alakazam','Machop','Machoke','Machamp','Bellsprout','Weepinbell','Victreebel','Tentacool','Tentacruel','Geodude','Graveler','Golem','Ponyta','Rapidash','Slowpoke','Slowbro','Magnemite','Magneton','Farfetch','Doduo','Dodrio','Seel','Dewgong','Grimer','Muk','Shellder','Cloyster','Gastly','Haunter','Gengar','Onix','Drowzee','Hypno','Krabby','Kingler','Voltorb','Electrode','Exeggcute','Exeggutor','Cubone','Marowak','Hitmonlee','Hitmonchan','Lickitung','Koffing','Weezing','Rhyhorn','Rhydon','Chansey','Tangela','Kangaskhan','Horsea','Seadra','Goldeen','Seaking','Staryu','Starmie','Mr. Mime','Scyther','Jynx','Electabuzz','Magmar','Pinsir','Tauros','Magikarp','Gyarados','Lapras','Ditto','Eevee','Vaporeon','Jolteon','Flareon','Porygon','Omanyte','Omastar','Kabuto','Kabutops','Aerodactyl','Snorlax','Articuno','Zapdos','Moltres','Dratini','Dragonair','Dragonite','Mewtwo','Mew']
ID=[]
for i in range(1,152):
    ID.append(i)
    
Pokemon_configuration=zip(ID,Names)   
Pokemon_configuration= dict (Pokemon_configuration)

dataset['names']=dataset['class'].map(Pokemon_configuration)   #associating pakemon names in the dataset
#print dataset.iloc[:,-1]

''' FIFTH PROBLEM SET '''

corr=dataset.corr()
c=corr['longitude'].sort_values(ascending=False)
#correlation function showing that Appeared hour has most impact on longitude and latitude, hence it will also come under prediction

Xnames=dataset.iloc[:,-1].values    #names column
ylat= dataset.iloc[:,1:2].values   #longitude column
ylong= dataset.iloc[:,2:3].values    #latitude column
X=dataset.iloc[:,13:14].values      #Appeared hour column
                  
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()          
Encoded_names=le.fit_transform(Xnames)
Encoded_names=np.asarray(Encoded_names)
Encoded_names=Encoded_names.reshape(len(Encoded_names),1)

"""from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse=False)
Encoded_names=enc.fit(Encoded_names)"""

"""
from sklearn.model_selection import train_test_split

Xnames_train,Xnames_test,ylong_train,ylong_test= train_test_split(Encoded_names,ylong,test_size= 0.2,random_state=0)

Xnames_train,Xnames_test,ylat_train,ylat_test= train_test_split(Encoded_names,ylat,test_size= 0.2,random_state=0)
"""

from sklearn.linear_model import LinearRegression

#Regression function for Appeared hour and Encoded names
reg= LinearRegression()
reg.fit(Encoded_names,X)  

pred1= reg.predict(Encoded_names) 


#regression function for longitude
reg_long = LinearRegression()
reg_long.fit(pred1,ylong)  

ylong_pred= reg_long.predict(pred1) 

#regression function for lattitude
reg_lat = LinearRegression()
reg_lat.fit(pred1,ylat) 

ylat_pred= reg_lat.predict(pred1)

n=raw_input("Enter name of pokemon whose location to be found(with proper casing):")

z=zip(Xnames,Encoded_names)   
r= dict (z)  #Dictionary containing pokemon names and their encoded value


print "Latitude for",n,"is:",reg_lat.predict(reg.predict(r[n])) 
print "Longitude for",n,"is:",reg_long.predict(reg.predict(r[n])) 


#importing library for ploting on map                  
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='merc',llcrnrlat=-55,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

# Coloring the continents
m.fillcontinents(color='grey',lake_color='grey')

#drawing boundary for countries
m.drawcountries(linewidth=0.1)

# fill in the oceans
m.drawmapboundary(fill_color='black')

#scattering the pokemon
m.scatter(reg_long.predict(reg.predict(r[n])),reg_lat.predict(reg.predict(r[n])),latlon=True,marker='.',color='blue',zorder=1)
plt.title("scattering pokemon")
plt.show()
