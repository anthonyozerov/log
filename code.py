#imports
from datetime import date, timedelta, datetime
from dateutil.parser import parse

import sys

from netCDF4 import Dataset
from ftplib import FTP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #used to create legends in plots
from mpl_toolkits.basemap import Basemap

#import data
log_pd = pd.read_csv("/home/aozerov/Dropbox/programming/jupyter/log_analysis/log_analysis_db3.csv")

#delete unneeded columns
coldel = ['Unnamed: 8']
log_pd = log_pd.drop(columns = coldel)

#remove empty rows
log_pd = log_pd[np.isfinite(log_pd['spentint'])]



#sleep pie chart
sleeps = []
for i in range (0, log_pd.shape[0]):
    if (log_pd.iloc[i]['Activity'] == 'sleep'):
        sleeps.append([log_pd.iloc[i]['spentint'],log_pd.iloc[i]['detail']])
npsleeps = np.array(sleeps)

unique, counts = np.unique(npsleeps[:,1], return_counts=True)
print(np.asarray((unique, counts)).T)

labels = ['Alarm','No alarm', 'No data']
sizes = [113,60,19]
colors = ['lightcoral', 'yellowgreen', 'lightgray']

plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=180)

plt.axis('equal')
plt.savefig("sleepalarmprops.png", dpi=300, transparent=True)
plt.show()



#sleep histogram (note: this code is quite poor, the code for the transit histogram is much better)
alarmweights = np.ones_like(alarmsleeplengths) / (len(alarmsleeplengths))
noalarmweights = np.ones_like(noalarmsleeplengths) / (len(noalarmsleeplengths))
nodataweights = np.ones_like(nodatasleeplengths) / (len(nodatasleeplengths))

red_patch = mpatches.Patch(color='red', label='Alarm')
green_patch = mpatches.Patch(color='green', label='No alarm')
blue_patch = mpatches.Patch(color='blue', label='No data')
plt.legend(handles=[red_patch,green_patch,blue_patch],prop={'size':10})

plt.hist(alarmsleeplengths, bins=binslocs,alpha=0.5, color = 'red', weights = alarmweights * 113/189)
# last part adjusts for probability of alarm so that it reflects probability of that length of sleep and an alarm
plt.hist(noalarmsleeplengths, bins=binslocs,alpha=0.5, color = 'green', weights = noalarmweights * 60/189)
plt.hist(nodatasleeplengths, bins=binslocs,alpha=0.5, color = 'blue', weights = nodataweights * 19/189)
plt.savefig("sleepalarms.png", dpi=300, transparent=True)
plt.show()



#transit pie chart
relocations = []
for i in range (0, log_pd.shape[0]):
    if (log_pd.iloc[i]['Activity'] == 'relocation'):
        relocations.append([log_pd.iloc[i]['spentint'],log_pd.iloc[i]['detail']])  
nprelocations = np.array(relocations)

labels = ['Car','Ferry', 'Plane','Public Transit','Schoolbus','Taxi','Walk']
sizes = counts

cmap = plt.get_cmap('Pastel2')
colors = cmap(np.linspace(0, 1, len(labels)))

plt.pie(sizes, labels=None, colors=colors,
        autopct=None, shadow=False, startangle=150, rotatelabels=True)

legends = []
for i in range (0, len(labels)):
    new_patch = mpatches.Patch(color=colors[i], label=labels[i])
    legends.append(new_patch)

plt.axis('equal')
plt.legend(handles=legends, prop={'size':10})
plt.savefig("relocations.png", dpi=300, transparent=True)
plt.show()



#transit histogram
binslocs = []
for i in range (0, 40):
    binslocs.append(i*0.075)

weights=[]
binned=[]
colorarray=[]
for i in range (0, len(unique)):
    transportlengths = []
    for k in range(0, log_pd.shape[0]):
        if (log_pd.iloc[k]['Activity'] == 'relocation' and log_pd.iloc[k]['detail'] == unique[i]):
            transportlengths.append(log_pd.iloc[k]['spentint'] * 24)
    transportweights = np.ones_like(transportlengths) / (len(transportlengths))
    binned.append(transportlengths)
    weights.append(transportweights * sizes[i]/sum(sizes))
    colorarray.append(colors[i])
#    plt.hist(transportlengths, bins=binslocs,alpha=0.5, color = colors[i], weights = transportweights * sizes[i]/sum(sizes))
plt.hist(binned, bins=binslocs, color = colorarray, weights = weights, stacked=True)
plt.savefig("relocations_hist.png", dpi=300, transparent=True)
plt.show()



#adjusted transit pie chart
relocations = []
for i in range (0, log_pd.shape[0]):
    if (log_pd.iloc[i]['Activity'] == 'relocation'):
        relocations.append(log_pd.iloc[i]['detail'])  
nprelocations = np.array(relocations)
unique = np.unique(nprelocations)

relocationsums = []
for i in range (0, len(unique)):
    relocationsum = 0
    for k in range (0, log_pd.shape[0]):
        if (log_pd.iloc[k]['detail'] == unique[i]):
            relocationsum += (log_pd.iloc[k]['spentint'])
    relocationsums.append(relocationsum)


labels = ['Car','Ferry', 'Plane','Public Transit','Schoolbus','Taxi','Walk']
sizes = relocationsums

cmap = plt.get_cmap('Pastel2')
colors = cmap(np.linspace(0, 1, len(unique)))

plt.pie(sizes, labels=labels, colors=colors,
        autopct=None, shadow=False, startangle=150, rotatelabels=True, pctdistance = 0.8,)

legends = []
for i in range (0, len(labels)):
    new_patch = mpatches.Patch(color=colors[i], label=labels[i])
    legends.append(new_patch)

plt.axis('equal')
#plt.legend(handles=legends, prop={'size':10})
plt.savefig("relocationsadjusted.png", dpi=300, transparent=True, bbox_inches="tight")
plt.show()



#cumulative sums of some school-related activities
activities = ['class','schoolwork','colappwork','work','ec','test']
labels = []
labeldict = {}
cmap = plt.get_cmap('Pastel2')
colors = cmap(np.linspace(0, 1, len(activities)))

plt.figure(dpi=600)
for i in range (0, len(activities)):
    new_patch = mpatches.Patch(color=colors[i], label=activities[i])
    labels.append(new_patch)
    x = [datetime(2018,4,18,19,2,0)]
    y = [0]
    for k in range (0, log_pd.shape[0]-1):
        if (log_pd.iloc[k]['Activity'] == activities[i] or log_pd.iloc[k]['by;with;for;about'] == activities[i] or log_pd.iloc[k]['detail'] == activities[i]):
            datetimestr = log_pd.iloc[k]['Start date/time (UTC)']
            datetimeobj = parse(datetimestr)
            x.append(datetimeobj)
            y.append(0)
            datetimestr = log_pd.iloc[k+1]['Start date/time (UTC)']
            datetimeobj = parse(datetimestr)
            x.append(datetimeobj)
            y.append(log_pd.iloc[k]['spentint'].astype(np.float))
    x.append(datetime(2018,10,20,16,54,38))
    ycumsum = np.array(y).cumsum()
    yprop = []
    for k in range (0, len(ycumsum)):
        yprop.append(ycumsum[k]/ycumsum[len(ycumsum)-1])
        
    yprop.append(1)
    
    plt.plot(x,yprop,'-',color = colors[i], linewidth = 1)


plt.legend(handles=labels, prop={'size':6})
x = [(parse(log_pd.iloc[0]['Start date/time (UTC)'])),(parse(log_pd.iloc[log_pd.shape[0]-1]['Start date/time (UTC)']))]
y = [0,1]
plt.plot(x, y,'--',linewidth = 0.5, color = "black")
plt.axvline(x=datetime(2018,6,20,8,45,16), linestyle='--', ymin=0, ymax = 1, linewidth=1, color='black')
plt.axvline(x=datetime(2018,8,21,5,18,56), linestyle='--', ymin=0, ymax = 1, linewidth=1, color='black')
plt.suptitle('Cumulative sums')
plt.savefig("schoolworkcumsums.png", dpi=300, transparent=True)
plt.show()



#cumulative sums of some non-school-related activities
activities = ['sleep','food','hygiene','log','language','relocation',';raina','vg']
labels = []
labeldict = {}
cmap = plt.get_cmap('Pastel2')
colors = cmap(np.linspace(0, 1, len(activities)))

plt.figure(dpi=600)
for i in range (0, len(activities)):
    labeldictkey = activities[i]
    new_patch = mpatches.Patch(color=colors[i], label=activities[i])
    labels.append(new_patch)
    x = [datetime(2018,4,18,19,2,0)]
    y = [0]
    for k in range (0, log_pd.shape[0]-1):
        if (log_pd.iloc[k]['Activity'] == activities[i] or log_pd.iloc[k]['by;with;for;about'] == activities[i]):
            datetimestr = log_pd.iloc[k]['Start date/time (UTC)']
            datetimeobj = parse(datetimestr)
            x.append(datetimeobj)
            y.append(0)
            datetimestr = log_pd.iloc[k+1]['Start date/time (UTC)']
            datetimeobj = parse(datetimestr)
            x.append(datetimeobj)
            y.append(log_pd.iloc[k]['spentint'].astype(np.float))
    x.append(parse(log_pd.iloc[log_pd.shape[0]-1]['Start date/time (UTC)']))
    ycumsum = np.array(y).cumsum()
    yprop = []
    for k in range (0, len(ycumsum)):
        yprop.append(ycumsum[k]/ycumsum[len(ycumsum)-1])
        
    yprop.append(1)
    plt.plot(x,yprop,'-',color = colors[i], linewidth = 1)


plt.legend(handles=labels, prop={'size':6})
x = [(parse(log_pd.iloc[0]['Start date/time (UTC)'])),(parse(log_pd.iloc[log_pd.shape[0]-1]['Start date/time (UTC)']))]
y = [0,1]
plt.plot(x, y,'--',linewidth = 0.5, color = "black")
plt.axvline(x=datetime(2018,6,20,8,45,16), linestyle='--', ymin=0, ymax = 1, linewidth=1, color='black')
plt.axvline(x=datetime(2018,8,21,5,18,56), linestyle='--', ymin=0, ymax = 1, linewidth=1, color='black')
plt.suptitle('Cumulative sums')
plt.savefig("nonschoolcumsums.png", dpi=300, transparent=True)
plt.show()



#orhographic globe projection
map = Basemap(projection='ortho',lat_0=45,lon_0=-20,resolution='l')

map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.fillcontinents(color='palegreen',lake_color='lightblue', alpha = 0.5)

map.drawmapboundary(fill_color='lightblue')

map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))

msklon = 37.6
msklat = 55.75
map.drawgreatcircle(-74,40.75,msklon,msklat,linewidth=0.5,color='red') #ny-msk
map.drawgreatcircle(24.93,60.169,msklon,msklat,linewidth=0.5,color='red') #hel-msk
map.drawgreatcircle(16.372,48.208,msklon,msklat,linewidth=0.5,color='red') #vienna-msk
map.drawgreatcircle(17.106,48.148,16.372,48.208,linewidth=0.5,color='red') #bratislava-vienna
map.drawgreatcircle(-74,40.75,-74.8478298,46.2229071,linewidth=0.5,color='red') #ny-laclabelle
map.drawgreatcircle(-74,40.75,-75.16379,39.952,linewidth=0.5,color='red') #ny-philadelphia

plt.savefig("map.png", dpi=300, transparent=True)

plt.show()
