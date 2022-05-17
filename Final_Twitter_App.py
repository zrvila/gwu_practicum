# Importing Packages
import datetime
import pandas as pd
import re
import spacy
import time
import schedule
import bokeh as bk
from bokeh.io import curdoc
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, Button, TextInput, Div
import os
import shutil
from bokeh.palettes import all_palettes
from bokeh.layouts import gridplot, row, column, layout, grid
from bokeh.models.widgets import Select, Tabs, Panel
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation
import collections
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from textblob import TextBlob
import sys
import os
import nltk
import pycountry
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import CountVectorizer
from math import pi
from bokeh.palettes import Category10
from bokeh.plotting import figure, show
from bokeh.transform import cumsum
import datetime as dt
from datetime import datetime

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 5000000

########### Creating a SQL connection to our SQLite database ###########
con = sqlite3.connect("twitter.sqlite")

############################# A. ALL SOURCES #####################################################################################################

df_all = pd.read_sql('select * from twitter_table', con)
# Dataframe for Time Series Plot
df_all['Time'] = pd.to_datetime(df_all.Time, format='%Y-%m-%d %H:%M:%S')
df_all['Time'] = df_all['Time'].dt.date

########### Creating dataframes for each Theme ###########
Vaccines_all=df_all.Tweets.str.contains("vaccines?|vax+|jabs?|mrna|biontech|pfizer|moderna|J&J|Johnson\s?&\s?Johnson", flags=re.IGNORECASE)
Vaccines_all =df_all.loc[Vaccines_all]

Mandates_all=df_all.Tweets.str.contains("mandates?|mask mandates?|vaccine mandates?|vaccine cards?|passports?|lockdowns?|quarantines?|restrictions?", flags=re.IGNORECASE)
Mandates_all =df_all.loc[Mandates_all]

Alternative_all=df_all.Tweets.str.contains("vitamins?|zinc|ivermectin", flags=re.IGNORECASE)
Alternative_all =df_all.loc[Alternative_all]

Health_all=df_all.Tweets.str.contains("CDC|NIH|FDA|Centers for Disease Control|National Institutes of Health|Food and Drug Administration|World Health Organization", flags=re.IGNORECASE)|df_all.Tweets.str.contains("WHO")
Health_all =df_all.loc[Health_all]

Conspiracies_all=df_all.Tweets.str.contains("bioweapons?|labs?", flags=re.IGNORECASE)
Conspiracies_all =df_all.loc[Conspiracies_all]

Origin_all =df_all.Tweets.str.contains("Wuhan|China", flags=re.IGNORECASE)
Origin_all =df_all.loc[Origin_all]

#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

#creating distinct dfs
sentences_vac_all= [sentence for sentence in Vaccines_all.Tweets]
sentences_man_all = [sentence for sentence in Mandates_all.Tweets]
sentences_alt_all = [sentence for sentence in Alternative_all.Tweets]
sentences_hea_all = [sentence for sentence in Health_all.Tweets]
sentences_con_all = [sentence for sentence in Conspiracies_all.Tweets]
sentences_ori_all = [sentence for sentence in Origin_all.Tweets]
sentences_all = [sentence for sentence in df_all.Tweets]

############ 1a. Creating Vaccine Word Count dataframe ############
lines = []
for sentence in sentences_vac_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df12_all = pd.DataFrame(stem2, columns=['Word'])
df12_all= df12_all['Word'].value_counts()
df12_all = pd.DataFrame(df12_all)
df12_all['Theme'] = "Vaccines"

############ 1b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df12p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df12p_all = df12p_all.loc[(df12p_all['Entity'] == 'PERSON')]
df12p_all = df12p_all['Word'].value_counts()
df12p_all = pd.DataFrame(df12p_all)
df12p_all['Theme'] = "Vaccines"

############ 2a. Creating Mandates Word Count dataframe ############
lines = []
for sentence in sentences_man_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

#Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df13_all = pd.DataFrame(stem2, columns=['Word'])
df13_all = df13_all['Word'].value_counts()
df13_all = pd.DataFrame(df13_all)
df13_all['Theme'] = "Mandates"
df12_all=df12_all.append(df13_all)

############ 2b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df13p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df13p_all = df13p_all.loc[(df13p_all['Entity'] == 'PERSON')]
df13p_all = df13p_all['Word'].value_counts()
df13p_all = pd.DataFrame(df13p_all)
df13p_all['Theme'] = "Mandates"
df12p_all = df12p_all.append(df13p_all)

############ 3a. Creating Alternative Treatments Word Count dataframe ############
lines = []
for sentence in sentences_alt_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df14_all = pd.DataFrame(stem2, columns=['Word'])
df14_all = df14_all['Word'].value_counts()
df14_all = pd.DataFrame(df14_all)
df14_all['Theme'] = "Alternative Treatments"
df12_all=df12_all.append(df14_all)

############ 3b. Creating Alternative Treatments Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df14p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df14p_all = df14p_all.loc[(df14p_all['Entity'] == 'PERSON')]
df14p_all = df14p_all['Word'].value_counts()
df14p_all = pd.DataFrame(df14p_all)
df14p_all['Theme'] = "Alternative Treatments"
df12p_all = df12p_all.append(df14p_all)

############ 4a. Creating Health Organizations Word Count dataframe ############
lines = []
for sentence in sentences_hea_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df15_all = pd.DataFrame(stem2, columns=['Word'])
df15_all = df15_all['Word'].value_counts()
df15_all = pd.DataFrame(df15_all)
df15_all['Theme'] = "Health Organizations"
df12_all=df12_all.append(df15_all)

############ 4b. Creating Health Organizations Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df15p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df15p_all = df15p_all.loc[(df15p_all['Entity'] == 'PERSON')]
df15p_all = df15p_all['Word'].value_counts()
df15p_all = pd.DataFrame(df15p_all)
df15p_all['Theme'] = "Health Organizations"
df12p_all = df12p_all.append(df15p_all)

############ 5a. Creating Conspiracies Word Count dataframe ############
lines = []
for sentence in sentences_con_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df16_all = pd.DataFrame(stem2, columns=['Word'])
df16_all = df16_all['Word'].value_counts()
df16_all = pd.DataFrame(df16_all)
df16_all['Theme'] = "Conspiracies"
df12_all = df12_all.append(df16_all)

############ 5b. Creating Conspiracies Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df16p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df16p_all = df16p_all.loc[(df16p_all['Entity'] == 'PERSON')]
df16p_all = df16p_all['Word'].value_counts()
df16p_all = pd.DataFrame(df16p_all)
df16p_all['Theme'] = "Conspiracies"
df12p_all = df12p_all.append(df16p_all)

############ 6a. Creating Origin Word Count dataframe ############
lines = []
for sentence in sentences_ori_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df17_all = pd.DataFrame(stem2, columns=['Word'])
df17_all = df17_all['Word'].value_counts()
df17_all = pd.DataFrame(df17_all)
df17_all['Theme'] = "Origin"
df12_all=df12_all.append(df17_all)

############ 6b. Creating Origin Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df17p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df17p_all = df17p_all.loc[(df17p_all['Entity'] == 'PERSON')]
df17p_all = df17p_all['Word'].value_counts()
df17p_all = pd.DataFrame(df17p_all)
df17p_all['Theme'] = "Origin"
df12p_all = df12p_all.append(df17p_all)

############ 7a. Creating All Themes Word Count dataframe ############
lines = []
for sentence in sentences_all:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)
        
#word cloud data initialization
stem_word_cloud = stem2

#count number of words
df18_all = pd.DataFrame(stem2, columns=['Word'])
df18_all = df18_all['Word'].value_counts()
df18_all = pd.DataFrame(df18_all)
df18_all['Theme'] = "All Themes"
df12_all=df12_all.append(df18_all)

############ 7b. Creating All Themes Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df18p_all = pd.DataFrame(label, columns=['Word', 'Entity'])
df18p_all = df18p_all.loc[(df18p_all['Entity'] == 'PERSON')]
df18p_all = df18p_all['Word'].value_counts()
df18p_all = pd.DataFrame(df18p_all)
df18p_all['Theme'] = "All Themes"
df12p_all = df12p_all.append(df18p_all)

################### CREATING WORD COUNT BOKEH BAR CHART FOR ALL SOURCES a.k.a p1_all #########################

theme_default = 'All Themes'
df12_all_filt = df12_all.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
words_all = df12_all_filt.index.tolist()
counts_all = df12_all_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source1_all = ColumnDataSource(
    data=dict(words_all=words_all, counts_all=counts_all, color=pally))

# create a new plot with a title and axis labels
p1_all = figure(y_range=words_all,
            x_range=(0, max(counts_all)),
            title='Top Words Used in Tweets for All Sources',
            y_axis_label='Words in Tweets',
            x_axis_label='Word Count',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p1_all.hbar(y='words_all',
        right='counts_all',
        height=0.2,
        color='color',
        source=word_source1_all)
def update_p1_all(attrname, old, new):
    theme1_all = widget1_all.value
    df12_all_filt = df12_all.query("Theme =='"+theme1_all+"'").sort_values('Word',ascending=False).head(20)
    words_all = df12_all_filt.index.tolist()
    counts_all = df12_all_filt['Word'].tolist()
    word_source1_all.data=dict(words_all=words_all, counts_all=counts_all, color=pally)
    p1_all.y_range.factors=words_all

widget1_all = Select(title="Themes for Word Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget1_all.on_change('value', update_p1_all)

################### CREATING PERSON COUNT BOKEH BAR CHART FOR ALL SOURCE a.k.a p2_all #########################

theme_default = 'All Themes'
df12p_all_filt = df12p_all.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
people_all = df12p_all_filt.index.tolist()
counts2_all = df12p_all_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source2_all = ColumnDataSource(
    data=dict(people_all=people_all, counts2_all=counts2_all, color=pally))

# create a new plot with a title and axis labels
p2_all = figure(y_range=people_all,
            x_range=(0, max(counts2_all)),
            title='Top People Referenced in Tweets for All Sources',
            y_axis_label='Person',
            x_axis_label='References',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p2_all.hbar(y='people_all',
        right='counts2_all',
        height=0.2,
        color='color',
        source=word_source2_all)
def update_p2_all(attrname, old, new):
    theme2_all = widget2_all.value
    df12p_all_filt = df12p_all.query("Theme =='"+theme2_all+"'").sort_values('Word',ascending=False).head(20)
    people_all = df12p_all_filt.index.tolist()
    counts2_all = df12p_all_filt['Word'].tolist()
    word_source2_all.data=dict(people_all=people_all, counts2_all=counts2_all, color=pally)
    p2_all.y_range.factors=people_all
    p_all.children[0].children[1].children[1] = p2_all

widget2_all = Select(title="Themes for Person Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget2_all.on_change('value', update_p2_all)

################### CREATING TIMES SERIES CHART FOR ALL SOURCES a.k.a p3_all #########################

# prepare ts data
ts12_all = Vaccines_all.pivot_table(index='Time', aggfunc='count')
ts12_all["Theme"] = "Vaccines"

ts13_all = Mandates_all.pivot_table(index='Time', aggfunc='count')
ts13_all["Theme"] = "Mandates"
ts12_all = ts12_all.append(ts13_all)

ts14_all = Alternative_all.pivot_table(index='Time', aggfunc='count')
ts14_all["Theme"] = "Alternative Treatments"
ts12_all = ts12_all.append(ts14_all)

ts15_all = Health_all.pivot_table(index='Time', aggfunc='count')
ts15_all["Theme"] = "Health Organizations"
ts12_all = ts12_all.append(ts15_all)

ts16_all = Conspiracies_all.pivot_table(index='Time', aggfunc='count')
ts16_all["Theme"] = "Conspiracies"
ts12_all = ts12_all.append(ts16_all)

ts17_all = Origin_all.pivot_table(index='Time', aggfunc='count')
ts17_all["Theme"] = "Origin"
ts12_all = ts12_all.append(ts17_all)

ts18_all = df_all.pivot_table(index='Time', aggfunc='count')
ts18_all["Theme"] = "All Themes"
ts12_all = ts12_all.append(ts18_all)

theme_default = 'All Themes'
ts12_all_filt = ts12_all.query("Theme == '"+theme_default+"'")

ts12_all_filt.drop(ts12_all_filt.columns[1], axis=1, inplace=True)
dates_all = ts12_all_filt.index.tolist()
counts3_all = ts12_all_filt['Tweets'].tolist()
data_all = dict(dates_all=dates_all, counts3_all=counts3_all)
data_all = pd.DataFrame.from_dict(data_all)
data_all['dates_all'] = pd.to_datetime(data_all['dates_all'])

p3_all = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for All Sources Over Time")

p3_all.line(
        x="dates_all", y="counts3_all",
        line_width=0.5, line_color="dodgerblue",
        legend_label = "Tweets",
        source=data_all
        )

p3_all.xaxis.axis_label = 'Date'
p3_all.yaxis.axis_label = 'Tweets'

p3_all.legend.location = "top_left"

def update_p3_all(attrname, old, new):
    theme3_all = widget3_all.value
    ts12_all_filt = ts12_all.query("Theme == '"+theme3_all+"'")
    ts12_all_filt.drop(ts12_all_filt.columns[1], axis=1, inplace=True)
    dates_all = ts12_all_filt.index.tolist()
    counts3_all = ts12_all_filt['Tweets'].tolist()
    data_all = dict(dates_all=dates_all, counts3_all=counts3_all)
    data_all = pd.DataFrame.from_dict(data_all)
    data_all['dates_all'] = pd.to_datetime(data_all['dates_all'])
    p3_all = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for All Sources Over Time")
    p3_all.line(
    x="dates_all", y='counts3_all',
    line_width=0.5, line_color="dodgerblue",
    legend_label = "Total Tweets",
    source=data_all
        )
    p3_all.xaxis.axis_label = 'Date'
    p3_all.yaxis.axis_label = 'Number of Tweets'
    p3_all.legend.location = "top_left"
    p_all.children[1].children[0].children[1] = p3_all
    
widget3_all = Select(title="Themes for Times Series Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin']
)
widget3_all.on_change('value', update_p3_all)

################### CREATING SENTIMENT ANALYSIS PIE CHART FOR ALL SOURCES a.k.a p4_all #########################

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df_all['Subjectivity'] = df_all['Tweets'].apply(getTextSubjectivity)
df_all['Polarity'] = df_all['Tweets'].apply(getTextPolarity)

df_all = df_all.drop(df_all[df_all['Tweets'] == ''].index)

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

df_all['Score'] = df_all['Polarity'].apply(getTextAnalysis)

positive_all = df_all[df_all['Score'] == 'Positive']
negative_all = df_all[df_all['Score'] == 'Negative']
neutral_all = df_all[df_all['Score'] == 'Neutral']
values_all = int(negative_all.shape[0]/(df_all.shape[0])*100), int(neutral_all.shape[0]/(df_all.shape[0])*100), int(positive_all.shape[0]/(df_all.shape[0])*100)
labels_all = df_all.groupby('Score').count().index.values

pie_all = dict(zip(labels_all, values_all))

data_all = pd.Series(pie_all).reset_index(name='values_all').rename(columns={'index': 'labels_all'})
data_all['angle'] = data_all['values_all']/data_all['values_all'].sum() * 2*pi
data_all['color'] = Category10[len(pie_all)]

p4_all = figure(height=650, title="Sentiment Analysis for Tweets from All Sources",
           tools="hover", tooltips="@labels_all: @values_all%", x_range=(-0.5, 1.0))

p4_all.wedge(x=0, y=0, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='labels_all', source=data_all)

p4_all.axis.axis_label = None
p4_all.axis.visible = False
p4_all.grid.grid_line_color = None

################### CREATING K-MEANS CLUSTERING SCATTER PLOT FOR ALL SOURCES a.k.a p5_all #########################

#lowercase all text
df_all['Tweets'] = df_all['Tweets'].str.lower()

#define cluster words
mainstream_related_words = '''response safety quarantine guidance expert proven trial
patient statistic examine hypothesis scientist medical treatment science scientific
experiment metric recover evidence policy recommend advise advice research uncertainvary
study support clarify learn information official formal adapt accept identify findings 
insight develop maintain'''

fringe_related_words = '''weapon takeover sheep coup conspiracy plot collusion collude hidden secret 
motive rights mainstream fake hoax lie fraud force groom indoctrinate freedom triggered 
democrat-run values unleash resist corrupt setup agenda control power republican-run truth 
real woke'''

#process text
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
punctuation = list(
    string.punctuation)  #already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()


def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)


df_all.Tweets = df_all.Tweets.apply(furnished)

#save processed text to objects
mainstream = furnished(mainstream_related_words)
fringe = furnished(fringe_related_words)

#clean text for visual
string1 = mainstream
words = string1.split()
mainstream = " ".join(sorted(set(words), key=words.index))
string1 = fringe
words = string1.split()
fringe = " ".join(sorted(set(words), key=words.index))

#define scoring functions for text
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def get_scores(group, tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores


m_scores_all = get_scores(mainstream, df_all.Tweets.to_list())
f_scores_all = get_scores(fringe, df_all.Tweets.to_list())

# create a jaccard scored df_all.
data_all = {
    'names': df_all.Name.to_list(),
    'mainstream_score': m_scores_all,
    'fringe_score': f_scores_all
}
scores_df_all = pd.DataFrame(data_all)


#assign classes based on highest score
def get_classes(l1_all, l2_all):
    main_all = []
    frin_all = []
    for i, j, in zip(l1_all, l2_all):
        m = max(i, j)
        if m == i:
            main_all.append(1)
        else:
            main_all.append(0)
        if m == j:
            frin_all.append(1)
        else:
            frin_all.append(0)
            
    return main_all, frin_all


l1_all = scores_df_all.mainstream_score.to_list()
l2_all = scores_df_all.fringe_score.to_list()

main_all, frin_all = get_classes(l1_all, l2_all)
data_all = {'name': scores_df_all.names.to_list(), 'mainstream': main_all, 'fringe': frin_all}
class_df_all = pd.DataFrame(data_all)

#grouping the tweets by username
new_groups_df_all = class_df_all.groupby(['name']).sum()

#fitting kmeans to dataset
X_all = new_groups_df_all[['mainstream', 'fringe']].values

kmeans_all = KMeans(n_clusters=3)
kmeans_all.fit_predict(X_all)

# Bokeh plot
# Create a blank figure with necessary arguments
p5_all = figure(plot_width=800, plot_height=650,title='KMeans Tweets Clustering for All Sources')
p5_all.xaxis.axis_label = 'Number Tweets using Mainstream-like Words '
p5_all.yaxis.axis_label = 'Number of Tweets using Fringe-like Words'

clus_xs_all = []

clus_ys_all = []

for entry in kmeans_all.cluster_centers_:
    clus_xs_all.append(entry[0])
    clus_ys_all.append(entry[1])
    
p5_all.circle_cross(x=clus_xs_all, y=clus_ys_all, size=40, fill_alpha=0, line_width=2, color=['pink', 'red', 'purple'])
p5_all.text(text = ['Cluster 1', 'Cluster 2', 'Cluster 3'], x=clus_xs_all, y=clus_ys_all, text_font_size='30pt')
p5_all.x_range=Range1d(0,400)
p5_all.y_range=Range1d(0,400)

i = 0 #counter

for sample in X_all:
    if kmeans_all.labels_[i] == 0:
        p5_all.circle(x=sample[0], y=sample[1], size=15, color="pink")
    if kmeans_all.labels_[i] == 1:
        p5_all.circle(x=sample[0], y=sample[1], size=15, color="red")
    if kmeans_all.labels_[i] == 2:
        p5_all.circle(x=sample[0], y=sample[1], size=15, color="purple")
    i += 1

################### CREATING TAB LAYOUT FOR ALL SOURCES a.k.a p_all #########################

p_all = column(
    row(
        column(widget1_all, p1_all),
        column(widget2_all, p2_all),
        column(p4_all)),
    row(
        column(widget3_all, p3_all),
        column(p5_all))) 

############################# B. FRINGE SOURCES  #####################################################################################################

df_frin = df_all.groupby("Source Type")
df_frin = df_frin.get_group("Fringe")

########### Creating dataframes for each Theme ###########
Vaccines_frin=df_frin.Tweets.str.contains("vaccines?|vax+|jabs?|mrna|biontech|pfizer|moderna|J&J|Johnson\s?&\s?Johnson", flags=re.IGNORECASE)
Vaccines_frin =df_frin.loc[Vaccines_frin]

Mandates_frin=df_frin.Tweets.str.contains("mandates?|mask mandates?|vaccine mandates?|vaccine cards?|passports?|lockdowns?|quarantines?|restrictions?", flags=re.IGNORECASE)
Mandates_frin =df_frin.loc[Mandates_frin]

Alternative_frin=df_frin.Tweets.str.contains("vitamins?|zinc|ivermectin", flags=re.IGNORECASE)
Alternative_frin =df_frin.loc[Alternative_frin]

Health_frin=df_frin.Tweets.str.contains("CDC|NIH|FDA|Centers for Disease Control|National Institutes of Health|Food and Drug Administration|World Health Organization", flags=re.IGNORECASE)|df_frin.Tweets.str.contains("WHO")
Health_frin =df_frin.loc[Health_frin]

Conspiracies_frin=df_frin.Tweets.str.contains("bioweapons?|labs?", flags=re.IGNORECASE)
Conspiracies_frin =df_frin.loc[Conspiracies_frin]

Origin_frin =df_frin.Tweets.str.contains("Wuhan|China", flags=re.IGNORECASE)
Origin_frin =df_frin.loc[Origin_frin]

#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

#creating distinct dfs
sentences_vac_frin= [sentence for sentence in Vaccines_frin.Tweets]
sentences_man_frin = [sentence for sentence in Mandates_frin.Tweets]
sentences_alt_frin = [sentence for sentence in Alternative_frin.Tweets]
sentences_hea_frin = [sentence for sentence in Health_frin.Tweets]
sentences_con_frin = [sentence for sentence in Conspiracies_frin.Tweets]
sentences_ori_frin = [sentence for sentence in Origin_frin.Tweets]
sentences_frin = [sentence for sentence in df_frin.Tweets]

############ 1a. Creating Vaccine Word Count dataframe ############
lines = []
for sentence in sentences_vac_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df12_frin = pd.DataFrame(stem2, columns=['Word'])
df12_frin= df12_frin['Word'].value_counts()
df12_frin = pd.DataFrame(df12_frin)
df12_frin['Theme'] = "Vaccines"

############ 1b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df12p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df12p_frin = df12p_frin.loc[(df12p_frin['Entity'] == 'PERSON')]
df12p_frin = df12p_frin['Word'].value_counts()
df12p_frin = pd.DataFrame(df12p_frin)
df12p_frin['Theme'] = "Vaccines"

############ 2a. Creating Mandates Word Count dataframe ############
lines = []
for sentence in sentences_man_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

#Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df13_frin = pd.DataFrame(stem2, columns=['Word'])
df13_frin = df13_frin['Word'].value_counts()
df13_frin = pd.DataFrame(df13_frin)
df13_frin['Theme'] = "Mandates"
df12_frin=df12_frin.append(df13_frin)

############ 2b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df13p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df13p_frin = df13p_frin.loc[(df13p_frin['Entity'] == 'PERSON')]
df13p_frin = df13p_frin['Word'].value_counts()
df13p_frin = pd.DataFrame(df13p_frin)
df13p_frin['Theme'] = "Mandates"
df12p_frin = df12p_frin.append(df13p_frin)

############ 3a. Creating Alternative Treatments Word Count dataframe ############
lines = []
for sentence in sentences_alt_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df14_frin = pd.DataFrame(stem2, columns=['Word'])
df14_frin = df14_frin['Word'].value_counts()
df14_frin = pd.DataFrame(df14_frin)
df14_frin['Theme'] = "Alternative Treatments"
df12_frin=df12_frin.append(df14_frin)

############ 3b. Creating Alternative Treatments Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df14p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df14p_frin = df14p_frin.loc[(df14p_frin['Entity'] == 'PERSON')]
df14p_frin = df14p_frin['Word'].value_counts()
df14p_frin = pd.DataFrame(df14p_frin)
df14p_frin['Theme'] = "Alternative Treatments"
df12p_frin = df12p_frin.append(df14p_frin)

############ 4a. Creating Health Organizations Word Count dataframe ############
lines = []
for sentence in sentences_hea_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df15_frin = pd.DataFrame(stem2, columns=['Word'])
df15_frin = df15_frin['Word'].value_counts()
df15_frin = pd.DataFrame(df15_frin)
df15_frin['Theme'] = "Health Organizations"
df12_frin=df12_frin.append(df15_frin)

############ 4b. Creating Health Organizations Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df15p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df15p_frin = df15p_frin.loc[(df15p_frin['Entity'] == 'PERSON')]
df15p_frin = df15p_frin['Word'].value_counts()
df15p_frin = pd.DataFrame(df15p_frin)
df15p_frin['Theme'] = "Health Organizations"
df12p_frin = df12p_frin.append(df15p_frin)

############ 5a. Creating Conspiracies Word Count dataframe ############
lines = []
for sentence in sentences_con_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df16_frin = pd.DataFrame(stem2, columns=['Word'])
df16_frin = df16_frin['Word'].value_counts()
df16_frin = pd.DataFrame(df16_frin)
df16_frin['Theme'] = "Conspiracies"
df12_frin = df12_frin.append(df16_frin)

############ 5b. Creating Conspiracies Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df16p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df16p_frin = df16p_frin.loc[(df16p_frin['Entity'] == 'PERSON')]
df16p_frin = df16p_frin['Word'].value_counts()
df16p_frin = pd.DataFrame(df16p_frin)
df16p_frin['Theme'] = "Conspiracies"
df12p_frin = df12p_frin.append(df16p_frin)

############ 6a. Creating Origin Word Count dataframe ############
lines = []
for sentence in sentences_ori_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df17_frin = pd.DataFrame(stem2, columns=['Word'])
df17_frin = df17_frin['Word'].value_counts()
df17_frin = pd.DataFrame(df17_frin)
df17_frin['Theme'] = "Origin"
df12_frin=df12_frin.append(df17_frin)

############ 6b. Creating Origin Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df17p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df17p_frin = df17p_frin.loc[(df17p_frin['Entity'] == 'PERSON')]
df17p_frin = df17p_frin['Word'].value_counts()
df17p_frin = pd.DataFrame(df17p_frin)
df17p_frin['Theme'] = "Origin"
df12p_frin = df12p_frin.append(df17p_frin)

############ 7a. Creating All Themes Word Count dataframe ############
lines = []
for sentence in sentences_frin:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)
        
#word cloud data initialization
stem_word_cloud = stem2

#count number of words
df18_frin = pd.DataFrame(stem2, columns=['Word'])
df18_frin = df18_frin['Word'].value_counts()
df18_frin = pd.DataFrame(df18_frin)
df18_frin['Theme'] = "All Themes"
df12_frin=df12_frin.append(df18_frin)

############ 7b. Creating All Themes Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df18p_frin = pd.DataFrame(label, columns=['Word', 'Entity'])
df18p_frin = df18p_frin.loc[(df18p_frin['Entity'] == 'PERSON')]
df18p_frin = df18p_frin['Word'].value_counts()
df18p_frin = pd.DataFrame(df18p_frin)
df18p_frin['Theme'] = "All Themes"
df12p_frin = df12p_frin.append(df18p_frin)

################### CREATING WORD COUNT BOKEH BAR CHART FOR FRINGE SOURCES a.k.a p1_frin #########################

theme_default = 'All Themes'
df12_frin_filt = df12_frin.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
words_frin = df12_frin_filt.index.tolist()
counts_frin = df12_frin_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source1_frin = ColumnDataSource(
    data=dict(words_frin=words_frin, counts_frin=counts_frin, color=pally))

# create a new plot with a title and axis labels
p1_frin = figure(y_range=words_frin,
            x_range=(0, max(counts_frin)),
            title='Top Words Used in Tweets for Fringe Sources',
            y_axis_label='Words in Tweets',
            x_axis_label='Word Count',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p1_frin.hbar(y='words_frin',
        right='counts_frin',
        height=0.2,
        color='color',
        source=word_source1_frin)
def update_p1_frin(attrname, old, new):
    theme1_frin = widget1_frin.value
    df12_frin_filt = df12_frin.query("Theme =='"+theme1_frin+"'").sort_values('Word',ascending=False).head(20)
    words_frin = df12_frin_filt.index.tolist()
    counts_frin = df12_frin_filt['Word'].tolist()
    word_source1_frin.data=dict(words_frin=words_frin, counts_frin=counts_frin, color=pally)
    p1_frin.y_range.factors=words_frin

widget1_frin = Select(title="Themes for Word Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget1_frin.on_change('value', update_p1_frin)

################### CREATING PERSON COUNT BOKEH BAR CHART FOR FRINGE SOURCES a.k.a p2_frin #########################

theme_default = 'All Themes'
df12p_frin_filt = df12p_frin.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
people_frin = df12p_frin_filt.index.tolist()
counts2_frin = df12p_frin_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source2_frin = ColumnDataSource(
    data=dict(people_frin=people_frin, counts2_frin=counts2_frin, color=pally))

# create a new plot with a title and axis labels
p2_frin = figure(y_range=people_frin,
            x_range=(0, max(counts2_frin)),
            title='Top People Referenced in Tweets for Fringe Sources',
            y_axis_label='Person',
            x_axis_label='References',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p2_frin.hbar(y='people_frin',
        right='counts2_frin',
        height=0.2,
        color='color',
        source=word_source2_frin)
def update_p2_frin(attrname, old, new):
    theme2_frin = widget2_frin.value
    df12p_frin_filt = df12p_frin.query("Theme =='"+theme2_frin+"'").sort_values('Word',ascending=False).head(20)
    people_frin = df12p_frin_filt.index.tolist()
    counts2_frin = df12p_frin_filt['Word'].tolist()
    word_source2_frin.data=dict(people_frin=people_frin, counts2_frin=counts2_frin, color=pally)
    p2_frin.y_range.factors=people_frin
    p_frin.children[0].children[1].children[1] = p2_frin

widget2_frin = Select(title="Themes for Person Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget2_frin.on_change('value', update_p2_frin)

################### CREATING TIMES SERIES CHART FOR FRINGE SOURCES a.k.a p3_frin #########################

# prepare ts data
ts12_frin = Vaccines_frin.pivot_table(index='Time', aggfunc='count')
ts12_frin["Theme"] = "Vaccines"

ts13_frin = Mandates_frin.pivot_table(index='Time', aggfunc='count')
ts13_frin["Theme"] = "Mandates"
ts12_frin = ts12_frin.append(ts13_frin)

ts14_frin = Alternative_frin.pivot_table(index='Time', aggfunc='count')
ts14_frin["Theme"] = "Alternative Treatments"
ts12_frin = ts12_frin.append(ts14_frin)

ts15_frin = Health_frin.pivot_table(index='Time', aggfunc='count')
ts15_frin["Theme"] = "Health Organizations"
ts12_frin = ts12_frin.append(ts15_frin)

ts16_frin = Conspiracies_frin.pivot_table(index='Time', aggfunc='count')
ts16_frin["Theme"] = "Conspiracies"
ts12_frin = ts12_frin.append(ts16_frin)

ts17_frin = Origin_frin.pivot_table(index='Time', aggfunc='count')
ts17_frin["Theme"] = "Origin"
ts12_frin = ts12_frin.append(ts17_frin)

ts18_frin = df_frin.pivot_table(index='Time', aggfunc='count')
ts18_frin["Theme"] = "All Themes"
ts12_frin = ts12_frin.append(ts18_frin)

theme_default = 'All Themes'
ts12_frin_filt = ts12_frin.query("Theme == '"+theme_default+"'")

ts12_frin_filt.drop(ts12_frin_filt.columns[1], axis=1, inplace=True)
dates_frin = ts12_frin_filt.index.tolist()
counts3_frin = ts12_frin_filt['Tweets'].tolist()
data_frin = dict(dates_frin=dates_frin, counts3_frin=counts3_frin)
data_frin = pd.DataFrame.from_dict(data_frin)
data_frin['dates_frin'] = pd.to_datetime(data_frin['dates_frin'])

p3_frin = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Fringe Sources Over Time")

p3_frin.line(
        x="dates_frin", y="counts3_frin",
        line_width=0.5, line_color="dodgerblue",
        legend_label = "Tweets",
        source=data_frin
        )

p3_frin.xaxis.axis_label = 'Date'
p3_frin.yaxis.axis_label = 'Tweets'

p3_frin.legend.location = "top_left"

def update_p3_frin(attrname, old, new):
    theme3_frin = widget3_frin.value
    ts12_frin_filt = ts12_frin.query("Theme == '"+theme3_frin+"'")
    ts12_frin_filt.drop(ts12_frin_filt.columns[1], axis=1, inplace=True)
    dates_frin = ts12_frin_filt.index.tolist()
    counts3_frin = ts12_frin_filt['Tweets'].tolist()
    data_frin = dict(dates_frin=dates_frin, counts3_frin=counts3_frin)
    data_frin = pd.DataFrame.from_dict(data_frin)
    data_frin['dates_frin'] = pd.to_datetime(data_frin['dates_frin'])
    p3_frin = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Fringe Sources Over Time")
    p3_frin.line(
    x="dates_frin", y='counts3_frin',
    line_width=0.5, line_color="dodgerblue",
    legend_label = "Total Tweets",
    source=data_frin
        )
    p3_frin.xaxis.axis_label = 'Date'
    p3_frin.yaxis.axis_label = 'Number of Tweets'
    p3_frin.legend.location = "top_left"
    p_frin.children[1].children[0].children[1] = p3_frin
    
widget3_frin = Select(title="Themes for Times Series Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin']
)
widget3_frin.on_change('value', update_p3_frin)

################### CREATING SENTIMENT ANALYSIS PIE CHART FOR FRINGE SOURCES a.k.a p4_frin #########################

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df_frin['Subjectivity'] = df_frin['Tweets'].apply(getTextSubjectivity)
df_frin['Polarity'] = df_frin['Tweets'].apply(getTextPolarity)

df_frin = df_frin.drop(df_frin[df_frin['Tweets'] == ''].index)

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

df_frin['Score'] = df_frin['Polarity'].apply(getTextAnalysis)

positive_frin = df_frin[df_frin['Score'] == 'Positive']
negative_frin = df_frin[df_frin['Score'] == 'Negative']
neutral_frin = df_frin[df_frin['Score'] == 'Neutral']
values_frin = int(negative_frin.shape[0]/(df_frin.shape[0])*100), int(neutral_frin.shape[0]/(df_frin.shape[0])*100), int(positive_frin.shape[0]/(df_frin.shape[0])*100)
labels_frin = df_frin.groupby('Score').count().index.values

pie_frin = dict(zip(labels_frin, values_frin))

data_frin = pd.Series(pie_frin).reset_index(name='values_frin').rename(columns={'index': 'labels_frin'})
data_frin['angle'] = data_frin['values_frin']/data_frin['values_frin'].sum() * 2*pi
data_frin['color'] = Category10[len(pie_frin)]

p4_frin = figure(height=650, title="Sentiment Analysis for Tweets from Fringe Sources",
           tools="hover", tooltips="@labels_frin: @values_frin%", x_range=(-0.5, 1.0))

p4_frin.wedge(x=0, y=0, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='labels_frin', source=data_frin)

p4_frin.axis.axis_label = None
p4_frin.axis.visible = False
p4_frin.grid.grid_line_color = None

################### CREATING K-MEANS CLUSTERING SCATTER PLOT FOR FRINGE SOURCES a.k.a p5_frin #########################

#lowercase all text
df_frin['Tweets'] = df_frin['Tweets'].str.lower()

#define cluster words
mainstream_related_words = '''response safety quarantine guidance expert proven trial
patient statistic examine hypothesis scientist medical treatment science scientific
experiment metric recover evidence policy recommend advise advice research uncertainvary
study support clarify learn information official formal adapt accept identify findings 
insight develop maintain'''

fringe_related_words = '''weapon takeover sheep coup conspiracy plot collusion collude hidden secret 
motive rights mainstream fake hoax lie fraud force groom indoctrinate freedom triggered 
democrat-run values unleash resist corrupt setup agenda control power republican-run truth 
real woke'''

#process text
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
punctuation = list(
    string.punctuation)  #already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()


def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)


df_frin.Tweets = df_frin.Tweets.apply(furnished)

#save processed text to objects
mainstream = furnished(mainstream_related_words)
fringe = furnished(fringe_related_words)

#clean text for visual
string1 = mainstream
words = string1.split()
mainstream = " ".join(sorted(set(words), key=words.index))
string1 = fringe
words = string1.split()
fringe = " ".join(sorted(set(words), key=words.index))

#define scoring functions for text
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def get_scores(group, tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores


m_scores_frin = get_scores(mainstream, df_frin.Tweets.to_list())
f_scores_frin = get_scores(fringe, df_frin.Tweets.to_list())

# create a jaccard scored df_frin.
data_frin = {
    'names': df_frin.Name.to_list(),
    'mainstream_score': m_scores_frin,
    'fringe_score': f_scores_frin
}
scores_df_frin = pd.DataFrame(data_frin)


#assign classes based on highest score
def get_classes(l1_frin, l2_frin):
    main_frin = []
    frin_frin = []
    for i, j, in zip(l1_frin, l2_frin):
        m = max(i, j)
        if m == i:
            main_frin.append(1)
        else:
            main_frin.append(0)
        if m == j:
            frin_frin.append(1)
        else:
            frin_frin.append(0)
            
    return main_frin, frin_frin


l1_frin = scores_df_frin.mainstream_score.to_list()
l2_frin = scores_df_frin.fringe_score.to_list()

main_frin, frin_frin = get_classes(l1_frin, l2_frin)
data_frin = {'name': scores_df_frin.names.to_list(), 'mainstream': main_frin, 'fringe': frin_frin}
class_df_frin = pd.DataFrame(data_frin)

#grouping the tweets by username
new_groups_df_frin = class_df_frin.groupby(['name']).sum()

#fitting kmeans to dataset
X_frin = new_groups_df_frin[['mainstream', 'fringe']].values

kmeans_frin = KMeans(n_clusters=3)
kmeans_frin.fit_predict(X_frin)

# Bokeh plot
# Create a blank figure with necessary arguments
p5_frin = figure(plot_width=800, plot_height=650,title='KMeans Tweets Clustering for Fringe Sources')
p5_frin.xaxis.axis_label = 'Number Tweets using Mainstream-like Words '
p5_frin.yaxis.axis_label = 'Number of Tweets using Fringe-like Words'

clus_xs_frin = []

clus_ys_frin = []

for entry in kmeans_frin.cluster_centers_:
    clus_xs_frin.append(entry[0])
    clus_ys_frin.append(entry[1])
    
p5_frin.circle_cross(x=clus_xs_frin, y=clus_ys_frin, size=40, fill_alpha=0, line_width=2, color=['pink', 'red', 'purple'])
p5_frin.text(text = ['Cluster 1', 'Cluster 2', 'Cluster 3'], x=clus_xs_frin, y=clus_ys_frin, text_font_size='30pt')
p5_frin.x_range=Range1d(0,400)
p5_frin.y_range=Range1d(0,400)

i = 0 #counter

for sample in X_frin:
    if kmeans_frin.labels_[i] == 0:
        p5_frin.circle(x=sample[0], y=sample[1], size=15, color="pink")
    if kmeans_frin.labels_[i] == 1:
        p5_frin.circle(x=sample[0], y=sample[1], size=15, color="red")
    if kmeans_frin.labels_[i] == 2:
        p5_frin.circle(x=sample[0], y=sample[1], size=15, color="purple")
    i += 1

################### CREATING TAB LAYOUT FOR FRINGE SOURCES a.k.a p_frin #########################

p_frin = column(
    row(
        column(widget1_frin, p1_frin),
        column(widget2_frin, p2_frin),
        column(p4_frin)),
    row(
        column(widget3_frin, p3_frin),
        column(p5_frin))) 

############################# C. MAINSTREAM SOURCES  #####################################################################################################

df_main = df_all.groupby("Source Type")
df_main = df_main.get_group("Mainstream")

########### Creating dataframes for each Theme ###########
Vaccines_main=df_main.Tweets.str.contains("vaccines?|vax+|jabs?|mrna|biontech|pfizer|moderna|J&J|Johnson\s?&\s?Johnson", flags=re.IGNORECASE)
Vaccines_main =df_main.loc[Vaccines_main]

Mandates_main=df_main.Tweets.str.contains("mandates?|mask mandates?|vaccine mandates?|vaccine cards?|passports?|lockdowns?|quarantines?|restrictions?", flags=re.IGNORECASE)
Mandates_main =df_main.loc[Mandates_main]

Alternative_main=df_main.Tweets.str.contains("vitamins?|zinc|ivermectin", flags=re.IGNORECASE)
Alternative_main =df_main.loc[Alternative_main]

Health_main=df_main.Tweets.str.contains("CDC|NIH|FDA|Centers for Disease Control|National Institutes of Health|Food and Drug Administration|World Health Organization", flags=re.IGNORECASE)|df_main.Tweets.str.contains("WHO")
Health_main =df_main.loc[Health_main]

Conspiracies_main=df_main.Tweets.str.contains("bioweapons?|labs?", flags=re.IGNORECASE)
Conspiracies_main =df_main.loc[Conspiracies_main]

Origin_main =df_main.Tweets.str.contains("Wuhan|China", flags=re.IGNORECASE)
Origin_main =df_main.loc[Origin_main]

#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

#creating distinct dfs
sentences_vac_main= [sentence for sentence in Vaccines_main.Tweets]
sentences_man_main = [sentence for sentence in Mandates_main.Tweets]
sentences_alt_main = [sentence for sentence in Alternative_main.Tweets]
sentences_hea_main = [sentence for sentence in Health_main.Tweets]
sentences_con_main = [sentence for sentence in Conspiracies_main.Tweets]
sentences_ori_main = [sentence for sentence in Origin_main.Tweets]
sentences_main = [sentence for sentence in df_main.Tweets]

############ 1a. Creating Vaccine Word Count dataframe ############
lines = []
for sentence in sentences_vac_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df12_main = pd.DataFrame(stem2, columns=['Word'])
df12_main= df12_main['Word'].value_counts()
df12_main = pd.DataFrame(df12_main)
df12_main['Theme'] = "Vaccines"

############ 1b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df12p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df12p_main = df12p_main.loc[(df12p_main['Entity'] == 'PERSON')]
df12p_main = df12p_main['Word'].value_counts()
df12p_main = pd.DataFrame(df12p_main)
df12p_main['Theme'] = "Vaccines"

############ 2a. Creating Mandates Word Count dataframe ############
lines = []
for sentence in sentences_man_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

#Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df13_main = pd.DataFrame(stem2, columns=['Word'])
df13_main = df13_main['Word'].value_counts()
df13_main = pd.DataFrame(df13_main)
df13_main['Theme'] = "Mandates"
df12_main=df12_main.append(df13_main)

############ 2b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df13p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df13p_main = df13p_main.loc[(df13p_main['Entity'] == 'PERSON')]
df13p_main = df13p_main['Word'].value_counts()
df13p_main = pd.DataFrame(df13p_main)
df13p_main['Theme'] = "Mandates"
df12p_main = df12p_main.append(df13p_main)

############ 3a. Creating Alternative Treatments Word Count dataframe ############
lines = []
for sentence in sentences_alt_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df14_main = pd.DataFrame(stem2, columns=['Word'])
df14_main = df14_main['Word'].value_counts()
df14_main = pd.DataFrame(df14_main)
df14_main['Theme'] = "Alternative Treatments"
df12_main=df12_main.append(df14_main)

############ 3b. Creating Alternative Treatments Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df14p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df14p_main = df14p_main.loc[(df14p_main['Entity'] == 'PERSON')]
df14p_main = df14p_main['Word'].value_counts()
df14p_main = pd.DataFrame(df14p_main)
df14p_main['Theme'] = "Alternative Treatments"
df12p_main = df12p_main.append(df14p_main)

############ 4a. Creating Health Organizations Word Count dataframe ############
lines = []
for sentence in sentences_hea_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df15_main = pd.DataFrame(stem2, columns=['Word'])
df15_main = df15_main['Word'].value_counts()
df15_main = pd.DataFrame(df15_main)
df15_main['Theme'] = "Health Organizations"
df12_main=df12_main.append(df15_main)

############ 4b. Creating Health Organizations Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df15p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df15p_main = df15p_main.loc[(df15p_main['Entity'] == 'PERSON')]
df15p_main = df15p_main['Word'].value_counts()
df15p_main = pd.DataFrame(df15p_main)
df15p_main['Theme'] = "Health Organizations"
df12p_main = df12p_main.append(df15p_main)

############ 5a. Creating Conspiracies Word Count dataframe ############
lines = []
for sentence in sentences_con_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df16_main = pd.DataFrame(stem2, columns=['Word'])
df16_main = df16_main['Word'].value_counts()
df16_main = pd.DataFrame(df16_main)
df16_main['Theme'] = "Conspiracies"
df12_main = df12_main.append(df16_main)

############ 5b. Creating Conspiracies Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df16p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df16p_main = df16p_main.loc[(df16p_main['Entity'] == 'PERSON')]
df16p_main = df16p_main['Word'].value_counts()
df16p_main = pd.DataFrame(df16p_main)
df16p_main['Theme'] = "Conspiracies"
df12p_main = df12p_main.append(df16p_main)

############ 6a. Creating Origin Word Count dataframe ############
lines = []
for sentence in sentences_ori_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df17_main = pd.DataFrame(stem2, columns=['Word'])
df17_main = df17_main['Word'].value_counts()
df17_main = pd.DataFrame(df17_main)
df17_main['Theme'] = "Origin"
df12_main=df12_main.append(df17_main)

############ 6b. Creating Origin Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df17p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df17p_main = df17p_main.loc[(df17p_main['Entity'] == 'PERSON')]
df17p_main = df17p_main['Word'].value_counts()
df17p_main = pd.DataFrame(df17p_main)
df17p_main['Theme'] = "Origin"
df12p_main = df12p_main.append(df17p_main)

############ 7a. Creating All Themes Word Count dataframe ############
lines = []
for sentence in sentences_main:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)
        
#word cloud data initialization
stem_word_cloud = stem2

#count number of words
df18_main = pd.DataFrame(stem2, columns=['Word'])
df18_main = df18_main['Word'].value_counts()
df18_main = pd.DataFrame(df18_main)
df18_main['Theme'] = "All Themes"
df12_main=df12_main.append(df18_main)

############ 7b. Creating All Themes Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df18p_main = pd.DataFrame(label, columns=['Word', 'Entity'])
df18p_main = df18p_main.loc[(df18p_main['Entity'] == 'PERSON')]
df18p_main = df18p_main['Word'].value_counts()
df18p_main = pd.DataFrame(df18p_main)
df18p_main['Theme'] = "All Themes"
df12p_main = df12p_main.append(df18p_main)

################### CREATING WORD COUNT BOKEH BAR CHART FOR MAINSTREAM SOURCES a.k.a p1_main #########################

theme_default = 'All Themes'
df12_main_filt = df12_main.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
words_main = df12_main_filt.index.tolist()
counts_main = df12_main_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source1_main = ColumnDataSource(
    data=dict(words_main=words_main, counts_main=counts_main, color=pally))

# create a new plot with a title and axis labels
p1_main = figure(y_range=words_main,
            x_range=(0, max(counts_main)),
            title='Top Words Used in Tweets for Mainstream Sources',
            y_axis_label='Words in Tweets',
            x_axis_label='Word Count',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p1_main.hbar(y='words_main',
        right='counts_main',
        height=0.2,
        color='color',
        source=word_source1_main)
def update_p1_main(attrname, old, new):
    theme1_main = widget1_main.value
    df12_main_filt = df12_main.query("Theme =='"+theme1_main+"'").sort_values('Word',ascending=False).head(20)
    words_main = df12_main_filt.index.tolist()
    counts_main = df12_main_filt['Word'].tolist()
    word_source1_main.data=dict(words_main=words_main, counts_main=counts_main, color=pally)
    p1_main.y_range.factors=words_main

widget1_main = Select(title="Themes for Word Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget1_main.on_change('value', update_p1_main)

################### CREATING PERSON COUNT BOKEH BAR CHART FOR MAINSTREAM SOURCES a.k.a p2_main #########################

theme_default = 'All Themes'
df12p_main_filt = df12p_main.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
people_main = df12p_main_filt.index.tolist()
counts2_main = df12p_main_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source2_main = ColumnDataSource(
    data=dict(people_main=people_main, counts2_main=counts2_main, color=pally))

# create a new plot with a title and axis labels
p2_main = figure(y_range=people_main,
            x_range=(0, max(counts2_main)),
            title='Top People Referenced in Tweets for Mainstream Sources',
            y_axis_label='Person',
            x_axis_label='References',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p2_main.hbar(y='people_main',
        right='counts2_main',
        height=0.2,
        color='color',
        source=word_source2_main)
def update_p2_main(attrname, old, new):
    theme2_main = widget2_main.value
    df12p_main_filt = df12p_main.query("Theme =='"+theme2_main+"'").sort_values('Word',ascending=False).head(20)
    people_main = df12p_main_filt.index.tolist()
    counts2_main = df12p_main_filt['Word'].tolist()
    word_source2_main.data=dict(people_main=people_main, counts2_main=counts2_main, color=pally)
    p2_main.y_range.factors=people_main
    p_main.children[0].children[1].children[1] = p2_main

widget2_main = Select(title="Themes for Person Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget2_main.on_change('value', update_p2_main)

################### CREATING TIMES SERIES CHART FOR MAINSTREAM SOURCES a.k.a p3_main #########################

# prepare ts data
ts12_main = Vaccines_main.pivot_table(index='Time', aggfunc='count')
ts12_main["Theme"] = "Vaccines"

ts13_main = Mandates_main.pivot_table(index='Time', aggfunc='count')
ts13_main["Theme"] = "Mandates"
ts12_main = ts12_main.append(ts13_main)

ts14_main = Alternative_main.pivot_table(index='Time', aggfunc='count')
ts14_main["Theme"] = "Alternative Treatments"
ts12_main = ts12_main.append(ts14_main)

ts15_main = Health_main.pivot_table(index='Time', aggfunc='count')
ts15_main["Theme"] = "Health Organizations"
ts12_main = ts12_main.append(ts15_main)

ts16_main = Conspiracies_main.pivot_table(index='Time', aggfunc='count')
ts16_main["Theme"] = "Conspiracies"
ts12_main = ts12_main.append(ts16_main)

ts17_main = Origin_main.pivot_table(index='Time', aggfunc='count')
ts17_main["Theme"] = "Origin"
ts12_main = ts12_main.append(ts17_main)

ts18_main = df_main.pivot_table(index='Time', aggfunc='count')
ts18_main["Theme"] = "All Themes"
ts12_main = ts12_main.append(ts18_main)

theme_default = 'All Themes'
ts12_main_filt = ts12_main.query("Theme == '"+theme_default+"'")

ts12_main_filt.drop(ts12_main_filt.columns[1], axis=1, inplace=True)
dates_main = ts12_main_filt.index.tolist()
counts3_main = ts12_main_filt['Tweets'].tolist()
data_main = dict(dates_main=dates_main, counts3_main=counts3_main)
data_main = pd.DataFrame.from_dict(data_main)
data_main['dates_main'] = pd.to_datetime(data_main['dates_main'])

p3_main = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Mainstream Sources Over Time")

p3_main.line(
        x="dates_main", y="counts3_main",
        line_width=0.5, line_color="dodgerblue",
        legend_label = "Tweets",
        source=data_main
        )

p3_main.xaxis.axis_label = 'Date'
p3_main.yaxis.axis_label = 'Tweets'

p3_main.legend.location = "top_left"

def update_p3_main(attrname, old, new):
    theme3_main = widget3_main.value
    ts12_main_filt = ts12_main.query("Theme == '"+theme3_main+"'")
    ts12_main_filt.drop(ts12_main_filt.columns[1], axis=1, inplace=True)
    dates_main = ts12_main_filt.index.tolist()
    counts3_main = ts12_main_filt['Tweets'].tolist()
    data_main = dict(dates_main=dates_main, counts3_main=counts3_main)
    data_main = pd.DataFrame.from_dict(data_main)
    data_main['dates_main'] = pd.to_datetime(data_main['dates_main'])
    p3_main = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Mainstream Sources Over Time")
    p3_main.line(
    x="dates_main", y='counts3_main',
    line_width=0.5, line_color="dodgerblue",
    legend_label = "Total Tweets",
    source=data_main
        )
    p3_main.xaxis.axis_label = 'Date'
    p3_main.yaxis.axis_label = 'Number of Tweets'
    p3_main.legend.location = "top_left"
    p_main.children[1].children[0].children[1] = p3_main
    
widget3_main = Select(title="Themes for Times Series Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin']
)
widget3_main.on_change('value', update_p3_main)

################### CREATING SENTIMENT ANALYSIS PIE CHART FOR MAINSTREAM SOURCES a.k.a p4_main #########################

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df_main['Subjectivity'] = df_main['Tweets'].apply(getTextSubjectivity)
df_main['Polarity'] = df_main['Tweets'].apply(getTextPolarity)

df_main = df_main.drop(df_main[df_main['Tweets'] == ''].index)

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

df_main['Score'] = df_main['Polarity'].apply(getTextAnalysis)

positive_main = df_main[df_main['Score'] == 'Positive']
negative_main = df_main[df_main['Score'] == 'Negative']
neutral_main = df_main[df_main['Score'] == 'Neutral']
values_main = int(negative_main.shape[0]/(df_main.shape[0])*100), int(neutral_main.shape[0]/(df_main.shape[0])*100), int(positive_main.shape[0]/(df_main.shape[0])*100)
labels_main = df_main.groupby('Score').count().index.values

pie_main = dict(zip(labels_main, values_main))

data_main = pd.Series(pie_main).reset_index(name='values_main').rename(columns={'index': 'labels_main'})
data_main['angle'] = data_main['values_main']/data_main['values_main'].sum() * 2*pi
data_main['color'] = Category10[len(pie_main)]

p4_main = figure(height=650, title="Sentiment Analysis for Tweets from Mainstream Sources",
           tools="hover", tooltips="@labels_main: @values_main%", x_range=(-0.5, 1.0))

p4_main.wedge(x=0, y=0, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='labels_main', source=data_main)

p4_main.axis.axis_label = None
p4_main.axis.visible = False
p4_main.grid.grid_line_color = None

################### CREATING K-MEANS CLUSTERING SCATTER PLOT FOR MAINSTREAM SOURCES a.k.a p5_main #########################

#lowercase all text
df_main['Tweets'] = df_main['Tweets'].str.lower()

#define cluster words
mainstream_related_words = '''response safety quarantine guidance expert proven trial
patient statistic examine hypothesis scientist medical treatment science scientific
experiment metric recover evidence policy recommend advise advice research uncertainvary
study support clarify learn information official formal adapt accept identify findings 
insight develop maintain'''

fringe_related_words = '''weapon takeover sheep coup conspiracy plot collusion collude hidden secret 
motive rights mainstream fake hoax lie fraud force groom indoctrinate freedom triggered 
democrat-run values unleash resist corrupt setup agenda control power republican-run truth 
real woke'''

#process text
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
punctuation = list(
    string.punctuation)  #already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()


def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)


df_main.Tweets = df_main.Tweets.apply(furnished)

#save processed text to objects
mainstream = furnished(mainstream_related_words)
fringe = furnished(fringe_related_words)

#clean text for visual
string1 = mainstream
words = string1.split()
mainstream = " ".join(sorted(set(words), key=words.index))
string1 = fringe
words = string1.split()
fringe = " ".join(sorted(set(words), key=words.index))

#define scoring functions for text
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def get_scores(group, tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores


m_scores_main = get_scores(mainstream, df_main.Tweets.to_list())
f_scores_main = get_scores(fringe, df_main.Tweets.to_list())

# create a jaccard scored df_main.
data_main = {
    'names': df_main.Name.to_list(),
    'mainstream_score': m_scores_main,
    'fringe_score': f_scores_main
}
scores_df_main = pd.DataFrame(data_main)


#assign classes based on highest score
def get_classes(l1_main, l2_main):
    main_main = []
    frin_main = []
    for i, j, in zip(l1_main, l2_main):
        m = max(i, j)
        if m == i:
            main_main.append(1)
        else:
            main_main.append(0)
        if m == j:
            frin_main.append(1)
        else:
            frin_main.append(0)
            
    return main_main, frin_main


l1_main = scores_df_main.mainstream_score.to_list()
l2_main = scores_df_main.fringe_score.to_list()

main_main, frin_main = get_classes(l1_main, l2_main)
data_main = {'name': scores_df_main.names.to_list(), 'mainstream': main_main, 'fringe': frin_main}
class_df_main = pd.DataFrame(data_main)

#grouping the tweets by username
new_groups_df_main = class_df_main.groupby(['name']).sum()

#fitting kmeans to dataset
X_main = new_groups_df_main[['mainstream', 'fringe']].values

kmeans_main = KMeans(n_clusters=3)
kmeans_main.fit_predict(X_main)

# Bokeh plot
# Create a blank figure with necessary arguments
p5_main = figure(plot_width=800, plot_height=650,title='KMeans Tweets Clustering for Mainstream Sources')
p5_main.xaxis.axis_label = 'Number Tweets using Mainstream-like Words '
p5_main.yaxis.axis_label = 'Number of Tweets using Fringe-like Words'

clus_xs_main = []

clus_ys_main = []

for entry in kmeans_main.cluster_centers_:
    clus_xs_main.append(entry[0])
    clus_ys_main.append(entry[1])
    
p5_main.circle_cross(x=clus_xs_main, y=clus_ys_main, size=40, fill_alpha=0, line_width=2, color=['pink', 'red', 'purple'])
p5_main.text(text = ['Cluster 1', 'Cluster 2', 'Cluster 3'], x=clus_xs_main, y=clus_ys_main, text_font_size='30pt')
p5_main.x_range=Range1d(0,400)
p5_main.y_range=Range1d(0,400)

i = 0 #counter

for sample in X_main:
    if kmeans_main.labels_[i] == 0:
        p5_main.circle(x=sample[0], y=sample[1], size=15, color="pink")
    if kmeans_main.labels_[i] == 1:
        p5_main.circle(x=sample[0], y=sample[1], size=15, color="red")
    if kmeans_main.labels_[i] == 2:
        p5_main.circle(x=sample[0], y=sample[1], size=15, color="purple")
    i += 1

################### CREATING TAB LAYOUT FOR MAINSTREAM SOURCES a.k.a p_main #########################

p_main = column(
    row(
        column(widget1_main, p1_main),
        column(widget2_main, p2_main),
        column(p4_main)),
    row(
        column(widget3_main, p3_main),
        column(p5_main))) 

############################# D. POLITICIANS  #####################################################################################################


df_pol = df_all.groupby("Source Type")
df_pol = df_pol.get_group("Politicians")

########### Creating dataframes for each Theme ###########
Vaccines_pol=df_pol.Tweets.str.contains("vaccines?|vax+|jabs?|mrna|biontech|pfizer|moderna|J&J|Johnson\s?&\s?Johnson", flags=re.IGNORECASE)
Vaccines_pol =df_pol.loc[Vaccines_pol]

Mandates_pol=df_pol.Tweets.str.contains("mandates?|mask mandates?|vaccine mandates?|vaccine cards?|passports?|lockdowns?|quarantines?|restrictions?", flags=re.IGNORECASE)
Mandates_pol =df_pol.loc[Mandates_pol]

Alternative_pol=df_pol.Tweets.str.contains("vitamins?|zinc|ivermectin", flags=re.IGNORECASE)
Alternative_pol =df_pol.loc[Alternative_pol]

Health_pol=df_pol.Tweets.str.contains("CDC|NIH|FDA|Centers for Disease Control|National Institutes of Health|Food and Drug Administration|World Health Organization", flags=re.IGNORECASE)|df_pol.Tweets.str.contains("WHO")
Health_pol =df_pol.loc[Health_pol]

Conspiracies_pol=df_pol.Tweets.str.contains("bioweapons?|labs?", flags=re.IGNORECASE)
Conspiracies_pol =df_pol.loc[Conspiracies_pol]

Origin_pol =df_pol.Tweets.str.contains("Wuhan|China", flags=re.IGNORECASE)
Origin_pol =df_pol.loc[Origin_pol]

#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

#creating distinct dfs
sentences_vac_pol= [sentence for sentence in Vaccines_pol.Tweets]
sentences_man_pol = [sentence for sentence in Mandates_pol.Tweets]
sentences_alt_pol = [sentence for sentence in Alternative_pol.Tweets]
sentences_hea_pol = [sentence for sentence in Health_pol.Tweets]
sentences_con_pol = [sentence for sentence in Conspiracies_pol.Tweets]
sentences_ori_pol = [sentence for sentence in Origin_pol.Tweets]
sentences_pol = [sentence for sentence in df_pol.Tweets]

############ 1a. Creating Vaccine Word Count dataframe ############
lines = []
for sentence in sentences_vac_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df12_pol = pd.DataFrame(stem2, columns=['Word'])
df12_pol= df12_pol['Word'].value_counts()
df12_pol = pd.DataFrame(df12_pol)
df12_pol['Theme'] = "Vaccines"

############ 1b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df12p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df12p_pol = df12p_pol.loc[(df12p_pol['Entity'] == 'PERSON')]
df12p_pol = df12p_pol['Word'].value_counts()
df12p_pol = pd.DataFrame(df12p_pol)
df12p_pol['Theme'] = "Vaccines"

############ 2a. Creating Mandates Word Count dataframe ############
lines = []
for sentence in sentences_man_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

#Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df13_pol = pd.DataFrame(stem2, columns=['Word'])
df13_pol = df13_pol['Word'].value_counts()
df13_pol = pd.DataFrame(df13_pol)
df13_pol['Theme'] = "Mandates"
df12_pol=df12_pol.append(df13_pol)

############ 2b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df13p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df13p_pol = df13p_pol.loc[(df13p_pol['Entity'] == 'PERSON')]
df13p_pol = df13p_pol['Word'].value_counts()
df13p_pol = pd.DataFrame(df13p_pol)
df13p_pol['Theme'] = "Mandates"
df12p_pol = df12p_pol.append(df13p_pol)

############ 3a. Creating Alternative Treatments Word Count dataframe ############
lines = []
for sentence in sentences_alt_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df14_pol = pd.DataFrame(stem2, columns=['Word'])
df14_pol = df14_pol['Word'].value_counts()
df14_pol = pd.DataFrame(df14_pol)
df14_pol['Theme'] = "Alternative Treatments"
df12_pol=df12_pol.append(df14_pol)

############ 3b. Creating Alternative Treatments Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df14p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df14p_pol = df14p_pol.loc[(df14p_pol['Entity'] == 'PERSON')]
df14p_pol = df14p_pol['Word'].value_counts()
df14p_pol = pd.DataFrame(df14p_pol)
df14p_pol['Theme'] = "Alternative Treatments"
df12p_pol = df12p_pol.append(df14p_pol)

############ 4a. Creating Health Organizations Word Count dataframe ############
lines = []
for sentence in sentences_hea_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df15_pol = pd.DataFrame(stem2, columns=['Word'])
df15_pol = df15_pol['Word'].value_counts()
df15_pol = pd.DataFrame(df15_pol)
df15_pol['Theme'] = "Health Organizations"
df12_pol=df12_pol.append(df15_pol)

############ 4b. Creating Health Organizations Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df15p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df15p_pol = df15p_pol.loc[(df15p_pol['Entity'] == 'PERSON')]
df15p_pol = df15p_pol['Word'].value_counts()
df15p_pol = pd.DataFrame(df15p_pol)
df15p_pol['Theme'] = "Health Organizations"
df12p_pol = df12p_pol.append(df15p_pol)

############ 5a. Creating Conspiracies Word Count dataframe ############
lines = []
for sentence in sentences_con_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df16_pol = pd.DataFrame(stem2, columns=['Word'])
df16_pol = df16_pol['Word'].value_counts()
df16_pol = pd.DataFrame(df16_pol)
df16_pol['Theme'] = "Conspiracies"
df12_pol = df12_pol.append(df16_pol)

############ 5b. Creating Conspiracies Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df16p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df16p_pol = df16p_pol.loc[(df16p_pol['Entity'] == 'PERSON')]
df16p_pol = df16p_pol['Word'].value_counts()
df16p_pol = pd.DataFrame(df16p_pol)
df16p_pol['Theme'] = "Conspiracies"
df12p_pol = df12p_pol.append(df16p_pol)

############ 6a. Creating Origin Word Count dataframe ############
lines = []
for sentence in sentences_ori_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df17_pol = pd.DataFrame(stem2, columns=['Word'])
df17_pol = df17_pol['Word'].value_counts()
df17_pol = pd.DataFrame(df17_pol)
df17_pol['Theme'] = "Origin"
df12_pol=df12_pol.append(df17_pol)

############ 6b. Creating Origin Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df17p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df17p_pol = df17p_pol.loc[(df17p_pol['Entity'] == 'PERSON')]
df17p_pol = df17p_pol['Word'].value_counts()
df17p_pol = pd.DataFrame(df17p_pol)
df17p_pol['Theme'] = "Origin"
df12p_pol = df12p_pol.append(df17p_pol)

############ 7a. Creating All Themes Word Count dataframe ############
lines = []
for sentence in sentences_pol:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)
        
#word cloud data initialization
stem_word_cloud = stem2

#count number of words
df18_pol = pd.DataFrame(stem2, columns=['Word'])
df18_pol = df18_pol['Word'].value_counts()
df18_pol = pd.DataFrame(df18_pol)
df18_pol['Theme'] = "All Themes"
df12_pol=df12_pol.append(df18_pol)

############ 7b. Creating All Themes Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df18p_pol = pd.DataFrame(label, columns=['Word', 'Entity'])
df18p_pol = df18p_pol.loc[(df18p_pol['Entity'] == 'PERSON')]
df18p_pol = df18p_pol['Word'].value_counts()
df18p_pol = pd.DataFrame(df18p_pol)
df18p_pol['Theme'] = "All Themes"
df12p_pol = df12p_pol.append(df18p_pol)

################### CREATING WORD COUNT BOKEH BAR CHART FOR POLITICIANS a.k.a p1_pol #########################

theme_default = 'All Themes'
df12_pol_filt = df12_pol.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
words_pol = df12_pol_filt.index.tolist()
counts_pol = df12_pol_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source1_pol = ColumnDataSource(
    data=dict(words_pol=words_pol, counts_pol=counts_pol, color=pally))

# create a new plot with a title and axis labels
p1_pol = figure(y_range=words_pol,
            x_range=(0, max(counts_pol)),
            title='Top Words Used in Tweets for Politicians',
            y_axis_label='Words in Tweets',
            x_axis_label='Word Count',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p1_pol.hbar(y='words_pol',
        right='counts_pol',
        height=0.2,
        color='color',
        source=word_source1_pol)
def update_p1_pol(attrname, old, new):
    theme1_pol = widget1_pol.value
    df12_pol_filt = df12_pol.query("Theme =='"+theme1_pol+"'").sort_values('Word',ascending=False).head(20)
    words_pol = df12_pol_filt.index.tolist()
    counts_pol = df12_pol_filt['Word'].tolist()
    word_source1_pol.data=dict(words_pol=words_pol, counts_pol=counts_pol, color=pally)
    p1_pol.y_range.factors=words_pol

widget1_pol = Select(title="Themes for Word Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget1_pol.on_change('value', update_p1_pol)

################### CREATING PERSON COUNT BOKEH BAR CHART FOR POLITICIANS a.k.a p2_pol #########################

theme_default = 'All Themes'
df12p_pol_filt = df12p_pol.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
people_pol = df12p_pol_filt.index.tolist()
counts2_pol = df12p_pol_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source2_pol = ColumnDataSource(
    data=dict(people_pol=people_pol, counts2_pol=counts2_pol, color=pally))

# create a new plot with a title and axis labels
p2_pol = figure(y_range=people_pol,
            x_range=(0, max(counts2_pol)),
            title='Top People Referenced in Tweets for Politicians',
            y_axis_label='Person',
            x_axis_label='References',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p2_pol.hbar(y='people_pol',
        right='counts2_pol',
        height=0.2,
        color='color',
        source=word_source2_pol)
def update_p2_pol(attrname, old, new):
    theme2_pol = widget2_pol.value
    df12p_pol_filt = df12p_pol.query("Theme =='"+theme2_pol+"'").sort_values('Word',ascending=False).head(20)
    people_pol = df12p_pol_filt.index.tolist()
    counts2_pol = df12p_pol_filt['Word'].tolist()
    word_source2_pol.data=dict(people_pol=people_pol, counts2_pol=counts2_pol, color=pally)
    p2_pol.y_range.factors=people_pol
    p_pol.children[0].children[1].children[1] = p2_pol


widget2_pol = Select(title="Themes for Person Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget2_pol.on_change('value', update_p2_pol)

################### CREATING TIMES SERIES CHART FOR POLITICIANS a.k.a p3_pol #########################

# prepare ts data
ts12_pol = Vaccines_pol.pivot_table(index='Time', aggfunc='count')
ts12_pol["Theme"] = "Vaccines"

ts13_pol = Mandates_pol.pivot_table(index='Time', aggfunc='count')
ts13_pol["Theme"] = "Mandates"
ts12_pol = ts12_pol.append(ts13_pol)

ts14_pol = Alternative_pol.pivot_table(index='Time', aggfunc='count')
ts14_pol["Theme"] = "Alternative Treatments"
ts12_pol = ts12_pol.append(ts14_pol)

ts15_pol = Health_pol.pivot_table(index='Time', aggfunc='count')
ts15_pol["Theme"] = "Health Organizations"
ts12_pol = ts12_pol.append(ts15_pol)

ts16_pol = Conspiracies_pol.pivot_table(index='Time', aggfunc='count')
ts16_pol["Theme"] = "Conspiracies"
ts12_pol = ts12_pol.append(ts16_pol)

ts17_pol = Origin_pol.pivot_table(index='Time', aggfunc='count')
ts17_pol["Theme"] = "Origin"
ts12_pol = ts12_pol.append(ts17_pol)

ts18_pol = df_pol.pivot_table(index='Time', aggfunc='count')
ts18_pol["Theme"] = "All Themes"
ts12_pol = ts12_pol.append(ts18_pol)

theme_default = 'All Themes'
ts12_pol_filt = ts12_pol.query("Theme == '"+theme_default+"'")

ts12_pol_filt.drop(ts12_pol_filt.columns[1], axis=1, inplace=True)
dates_pol = ts12_pol_filt.index.tolist()
counts3_pol = ts12_pol_filt['Tweets'].tolist()
data_pol = dict(dates_pol=dates_pol, counts3_pol=counts3_pol)
data_pol = pd.DataFrame.from_dict(data_pol)
data_pol['dates_pol'] = pd.to_datetime(data_pol['dates_pol'])

p3_pol = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Politicians Over Time")

p3_pol.line(
        x="dates_pol", y="counts3_pol",
        line_width=0.5, line_color="dodgerblue",
        legend_label = "Tweets",
        source=data_pol
        )

p3_pol.xaxis.axis_label = 'Date'
p3_pol.yaxis.axis_label = 'Tweets'

p3_pol.legend.location = "top_left"

def update_p3_pol(attrname, old, new):
    theme3_pol = widget3_pol.value
    ts12_pol_filt = ts12_pol.query("Theme == '"+theme3_pol+"'")
    ts12_pol_filt.drop(ts12_pol_filt.columns[1], axis=1, inplace=True)
    dates_pol = ts12_pol_filt.index.tolist()
    counts3_pol = ts12_pol_filt['Tweets'].tolist()
    data_pol = dict(dates_pol=dates_pol, counts3_pol=counts3_pol)
    data_pol = pd.DataFrame.from_dict(data_pol)
    data_pol['dates_pol'] = pd.to_datetime(data_pol['dates_pol'])
    p3_pol = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Politicians Over Time")
    p3_pol.line(
    x="dates_pol", y='counts3_pol',
    line_width=0.5, line_color="dodgerblue",
    legend_label = "Total Tweets",
    source=data_pol
        )
    p3_pol.xaxis.axis_label = 'Date'
    p3_pol.yaxis.axis_label = 'Number of Tweets'
    p3_pol.legend.location = "top_left"
    p_pol.children[1].children[0].children[1] = p3_pol
    
widget3_pol = Select(title="Themes for Times Series Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin']
)
widget3_pol.on_change('value', update_p3_pol)

################### CREATING SENTIMENT ANALYSIS PIE CHART FOR POLITICIANS a.k.a p4_pol #########################

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df_pol['Subjectivity'] = df_pol['Tweets'].apply(getTextSubjectivity)
df_pol['Polarity'] = df_pol['Tweets'].apply(getTextPolarity)

df_pol = df_pol.drop(df_pol[df_pol['Tweets'] == ''].index)

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

df_pol['Score'] = df_pol['Polarity'].apply(getTextAnalysis)

positive_pol = df_pol[df_pol['Score'] == 'Positive']
negative_pol = df_pol[df_pol['Score'] == 'Negative']
neutral_pol = df_pol[df_pol['Score'] == 'Neutral']
values_pol = int(negative_pol.shape[0]/(df_pol.shape[0])*100), int(neutral_pol.shape[0]/(df_pol.shape[0])*100), int(positive_pol.shape[0]/(df_pol.shape[0])*100)
labels_pol = df_pol.groupby('Score').count().index.values

pie_pol = dict(zip(labels_pol, values_pol))

data_pol = pd.Series(pie_pol).reset_index(name='values_pol').rename(columns={'index': 'labels_pol'})
data_pol['angle'] = data_pol['values_pol']/data_pol['values_pol'].sum() * 2*pi
data_pol['color'] = Category10[len(pie_pol)]

p4_pol = figure(height=650, title="Sentiment Analysis for Tweets from Politicians",
           tools="hover", tooltips="@labels_pol: @values_pol%", x_range=(-0.5, 1.0))

p4_pol.wedge(x=0, y=0, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='labels_pol', source=data_pol)

p4_pol.axis.axis_label = None
p4_pol.axis.visible = False
p4_pol.grid.grid_line_color = None

################### CREATING K-MEANS CLUSTERING SCATTER PLOT FOR POLITICIANS a.k.a p5_pol #########################

#lowercase all text
df_pol['Tweets'] = df_pol['Tweets'].str.lower()

#define cluster words
mainstream_related_words = '''response safety quarantine guidance expert proven trial
patient statistic examine hypothesis scientist medical treatment science scientific
experiment metric recover evidence policy recommend advise advice research uncertainvary
study support clarify learn information official formal adapt accept identify findings 
insight develop maintain'''

fringe_related_words = '''weapon takeover sheep coup conspiracy plot collusion collude hidden secret 
motive rights mainstream fake hoax lie fraud force groom indoctrinate freedom triggered 
democrat-run values unleash resist corrupt setup agenda control power republican-run truth 
real woke'''

#process text
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
punctuation = list(
    string.punctuation)  #already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()


def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)


df_pol.Tweets = df_pol.Tweets.apply(furnished)

#save processed text to objects
mainstream = furnished(mainstream_related_words)
fringe = furnished(fringe_related_words)

#clean text for visual
string1 = mainstream
words = string1.split()
mainstream = " ".join(sorted(set(words), key=words.index))
string1 = fringe
words = string1.split()
fringe = " ".join(sorted(set(words), key=words.index))

#define scoring functions for text
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def get_scores(group, tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores


m_scores_pol = get_scores(mainstream, df_pol.Tweets.to_list())
f_scores_pol = get_scores(fringe, df_pol.Tweets.to_list())

# create a jaccard scored df_pol.
data_pol = {
    'names': df_pol.Name.to_list(),
    'mainstream_score': m_scores_pol,
    'fringe_score': f_scores_pol
}
scores_df_pol = pd.DataFrame(data_pol)


#assign classes based on highest score
def get_classes(l1_pol, l2_pol):
    main_pol = []
    frin_pol = []
    for i, j, in zip(l1_pol, l2_pol):
        m = max(i, j)
        if m == i:
            main_pol.append(1)
        else:
            main_pol.append(0)
        if m == j:
            frin_pol.append(1)
        else:
            frin_pol.append(0)
            
    return main_pol, frin_pol


l1_pol = scores_df_pol.mainstream_score.to_list()
l2_pol = scores_df_pol.fringe_score.to_list()

main_pol, frin_pol = get_classes(l1_pol, l2_pol)
data_pol = {'name': scores_df_pol.names.to_list(), 'mainstream': main_pol, 'fringe': frin_pol}
class_df_pol = pd.DataFrame(data_pol)

#grouping the tweets by username
new_groups_df_pol = class_df_pol.groupby(['name']).sum()

#fitting kmeans to dataset
X_pol = new_groups_df_pol[['mainstream', 'fringe']].values

kmeans_pol = KMeans(n_clusters=3)
kmeans_pol.fit_predict(X_pol)

# Bokeh plot
# Create a blank figure with necessary arguments
p5_pol = figure(plot_width=800, plot_height=650,title='KMeans Tweets Clustering for Politicians')
p5_pol.xaxis.axis_label = 'Number Tweets using Mainstream-like Words '
p5_pol.yaxis.axis_label = 'Number of Tweets using Fringe-like Words'

clus_xs_pol = []

clus_ys_pol = []

for entry in kmeans_pol.cluster_centers_:
    clus_xs_pol.append(entry[0])
    clus_ys_pol.append(entry[1])
    
p5_pol.circle_cross(x=clus_xs_pol, y=clus_ys_pol, size=40, fill_alpha=0, line_width=2, color=['pink', 'red', 'purple'])
p5_pol.text(text = ['Cluster 1', 'Cluster 2', 'Cluster 3'], x=clus_xs_pol, y=clus_ys_pol, text_font_size='30pt')
p5_pol.x_range=Range1d(0,400)
p5_pol.y_range=Range1d(0,400)

i = 0 #counter

for sample in X_pol:
    if kmeans_pol.labels_[i] == 0:
        p5_pol.circle(x=sample[0], y=sample[1], size=15, color="pink")
    if kmeans_pol.labels_[i] == 1:
        p5_pol.circle(x=sample[0], y=sample[1], size=15, color="red")
    if kmeans_pol.labels_[i] == 2:
        p5_pol.circle(x=sample[0], y=sample[1], size=15, color="purple")
    i += 1

################### CREATING TAB LAYOUT FOR POLITICIANS a.k.a p_pol #########################

p_pol = column(
    row(
        column(widget1_pol, p1_pol),
        column(widget2_pol, p2_pol),
        column(p4_pol)),
    row(
        column(widget3_pol, p3_pol),
        column(p5_pol))) 

############################# E. SCIENTIFIC SOURCES  #####################################################################################################

df_sci = df_all.groupby("Source Type")
df_sci = df_sci.get_group("Scientific")

########### Creating dataframes for each Theme ###########
Vaccines_sci=df_sci.Tweets.str.contains("vaccines?|vax+|jabs?|mrna|biontech|pfizer|moderna|J&J|Johnson\s?&\s?Johnson", flags=re.IGNORECASE)
Vaccines_sci =df_sci.loc[Vaccines_sci]

Mandates_sci=df_sci.Tweets.str.contains("mandates?|mask mandates?|vaccine mandates?|vaccine cards?|passports?|lockdowns?|quarantines?|restrictions?", flags=re.IGNORECASE)
Mandates_sci =df_sci.loc[Mandates_sci]

Alternative_sci=df_sci.Tweets.str.contains("vitamins?|zinc|ivermectin", flags=re.IGNORECASE)
Alternative_sci =df_sci.loc[Alternative_sci]

Health_sci=df_sci.Tweets.str.contains("CDC|NIH|FDA|Centers for Disease Control|National Institutes of Health|Food and Drug Administration|World Health Organization", flags=re.IGNORECASE)|df_sci.Tweets.str.contains("WHO")
Health_sci =df_sci.loc[Health_sci]

Conspiracies_sci=df_sci.Tweets.str.contains("bioweapons?|labs?", flags=re.IGNORECASE)
Conspiracies_sci =df_sci.loc[Conspiracies_sci]

Origin_sci =df_sci.Tweets.str.contains("Wuhan|China", flags=re.IGNORECASE)
Origin_sci =df_sci.loc[Origin_sci]

#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

#creating distinct dfs
sentences_vac_sci= [sentence for sentence in Vaccines_sci.Tweets]
sentences_man_sci = [sentence for sentence in Mandates_sci.Tweets]
sentences_alt_sci = [sentence for sentence in Alternative_sci.Tweets]
sentences_hea_sci = [sentence for sentence in Health_sci.Tweets]
sentences_con_sci = [sentence for sentence in Conspiracies_sci.Tweets]
sentences_ori_sci = [sentence for sentence in Origin_sci.Tweets]
sentences_sci = [sentence for sentence in df_sci.Tweets]

############ 1a. Creating Vaccine Word Count dataframe ############
lines = []
for sentence in sentences_vac_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df12_sci = pd.DataFrame(stem2, columns=['Word'])
df12_sci= df12_sci['Word'].value_counts()
df12_sci = pd.DataFrame(df12_sci)
df12_sci['Theme'] = "Vaccines"

############ 1b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df12p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df12p_sci = df12p_sci.loc[(df12p_sci['Entity'] == 'PERSON')]
df12p_sci = df12p_sci['Word'].value_counts()
df12p_sci = pd.DataFrame(df12p_sci)
df12p_sci['Theme'] = "Vaccines"

############ 2a. Creating Mandates Word Count dataframe ############
lines = []
for sentence in sentences_man_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

#Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df13_sci = pd.DataFrame(stem2, columns=['Word'])
df13_sci = df13_sci['Word'].value_counts()
df13_sci = pd.DataFrame(df13_sci)
df13_sci['Theme'] = "Mandates"
df12_sci=df12_sci.append(df13_sci)

############ 2b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df13p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df13p_sci = df13p_sci.loc[(df13p_sci['Entity'] == 'PERSON')]
df13p_sci = df13p_sci['Word'].value_counts()
df13p_sci = pd.DataFrame(df13p_sci)
df13p_sci['Theme'] = "Mandates"
df12p_sci = df12p_sci.append(df13p_sci)

############ 3a. Creating Alternative Treatments Word Count dataframe ############
lines = []
for sentence in sentences_alt_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df14_sci = pd.DataFrame(stem2, columns=['Word'])
df14_sci = df14_sci['Word'].value_counts()
df14_sci = pd.DataFrame(df14_sci)
df14_sci['Theme'] = "Alternative Treatments"
df12_sci=df12_sci.append(df14_sci)

############ 3b. Creating Alternative Treatments Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df14p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df14p_sci = df14p_sci.loc[(df14p_sci['Entity'] == 'PERSON')]
df14p_sci = df14p_sci['Word'].value_counts()
df14p_sci = pd.DataFrame(df14p_sci)
df14p_sci['Theme'] = "Alternative Treatments"
df12p_sci = df12p_sci.append(df14p_sci)

############ 4a. Creating Health Organizations Word Count dataframe ############
lines = []
for sentence in sentences_hea_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df15_sci = pd.DataFrame(stem2, columns=['Word'])
df15_sci = df15_sci['Word'].value_counts()
df15_sci = pd.DataFrame(df15_sci)
df15_sci['Theme'] = "Health Organizations"
df12_sci=df12_sci.append(df15_sci)

############ 4b. Creating Health Organizations Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df15p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df15p_sci = df15p_sci.loc[(df15p_sci['Entity'] == 'PERSON')]
df15p_sci = df15p_sci['Word'].value_counts()
df15p_sci = pd.DataFrame(df15p_sci)
df15p_sci['Theme'] = "Health Organizations"
df12p_sci = df12p_sci.append(df15p_sci)

############ 5a. Creating Conspiracies Word Count dataframe ############
lines = []
for sentence in sentences_con_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

#count number of words
df16_sci = pd.DataFrame(stem2, columns=['Word'])
df16_sci = df16_sci['Word'].value_counts()
df16_sci = pd.DataFrame(df16_sci)
df16_sci['Theme'] = "Conspiracies"
df12_sci = df12_sci.append(df16_sci)

############ 5b. Creating Conspiracies Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df16p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df16p_sci = df16p_sci.loc[(df16p_sci['Entity'] == 'PERSON')]
df16p_sci = df16p_sci['Word'].value_counts()
df16p_sci = pd.DataFrame(df16p_sci)
df16p_sci['Theme'] = "Conspiracies"
df12p_sci = df12p_sci.append(df16p_sci)

############ 6a. Creating Origin Word Count dataframe ############
lines = []
for sentence in sentences_ori_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)


#count number of words
df17_sci = pd.DataFrame(stem2, columns=['Word'])
df17_sci = df17_sci['Word'].value_counts()
df17_sci = pd.DataFrame(df17_sci)
df17_sci['Theme'] = "Origin"
df12_sci=df12_sci.append(df17_sci)

############ 6b. Creating Origin Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df17p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df17p_sci = df17p_sci.loc[(df17p_sci['Entity'] == 'PERSON')]
df17p_sci = df17p_sci['Word'].value_counts()
df17p_sci = pd.DataFrame(df17p_sci)
df17p_sci['Theme'] = "Origin"
df12p_sci = df12p_sci.append(df17p_sci)

############ 7a. Creating All Themes Word Count dataframe ############
lines = []
for sentence in sentences_sci:
    words = sentence.split()
    for w in words:
        lines.append(w)

    #Removing Punctuation
lines = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) and re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)
        
#word cloud data initialization
stem_word_cloud = stem2

#count number of words
df18_sci = pd.DataFrame(stem2, columns=['Word'])
df18_sci = df18_sci['Word'].value_counts()
df18_sci = pd.DataFrame(df18_sci)
df18_sci['Theme'] = "All Themes"
df12_sci=df12_sci.append(df18_sci)

############ 7b. Creating All Themes Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
df18p_sci = pd.DataFrame(label, columns=['Word', 'Entity'])
df18p_sci = df18p_sci.loc[(df18p_sci['Entity'] == 'PERSON')]
df18p_sci = df18p_sci['Word'].value_counts()
df18p_sci = pd.DataFrame(df18p_sci)
df18p_sci['Theme'] = "All Themes"
df12p_sci = df12p_sci.append(df18p_sci)

################### CREATING WORD COUNT BOKEH BAR CHART FOR SCIENTIFIC SOURCES a.k.a p1_sci #########################

theme_default = 'All Themes'
df12_sci_filt = df12_sci.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
words_sci = df12_sci_filt.index.tolist()
counts_sci = df12_sci_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source1_sci = ColumnDataSource(
    data=dict(words_sci=words_sci, counts_sci=counts_sci, color=pally))

# create a new plot with a title and axis labels
p1_sci = figure(y_range=words_sci,
            x_range=(0, max(counts_sci)),
            title='Top Words Used in Tweets for Scientific Sources',
            y_axis_label='Words in Tweets',
            x_axis_label='Word Count',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p1_sci.hbar(y='words_sci',
        right='counts_sci',
        height=0.2,
        color='color',
        source=word_source1_sci)
def update_p1_sci(attrname, old, new):
    theme1_sci = widget1_sci.value
    df12_sci_filt = df12_sci.query("Theme =='"+theme1_sci+"'").sort_values('Word',ascending=False).head(20)
    words_sci = df12_sci_filt.index.tolist()
    counts_sci = df12_sci_filt['Word'].tolist()
    word_source1_sci.data=dict(words_sci=words_sci, counts_sci=counts_sci, color=pally)
    p1_sci.y_range.factors=words_sci

widget1_sci = Select(title="Themes for Word Bar Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin'])
widget1_sci.on_change('value', update_p1_sci)


################### CREATING TIMES SERIES CHART FOR SCIENTIFIC SOURCES a.k.a p3_sci #########################

# prepare ts data
ts12_sci = Vaccines_sci.pivot_table(index='Time', aggfunc='count')
ts12_sci["Theme"] = "Vaccines"

ts13_sci = Mandates_sci.pivot_table(index='Time', aggfunc='count')
ts13_sci["Theme"] = "Mandates"
ts12_sci = ts12_sci.append(ts13_sci)

ts14_sci = Alternative_sci.pivot_table(index='Time', aggfunc='count')
ts14_sci["Theme"] = "Alternative Treatments"
ts12_sci = ts12_sci.append(ts14_sci)

ts15_sci = Health_sci.pivot_table(index='Time', aggfunc='count')
ts15_sci["Theme"] = "Health Organizations"
ts12_sci = ts12_sci.append(ts15_sci)

ts16_sci = Conspiracies_sci.pivot_table(index='Time', aggfunc='count')
ts16_sci["Theme"] = "Conspiracies"
ts12_sci = ts12_sci.append(ts16_sci)

ts17_sci = Origin_sci.pivot_table(index='Time', aggfunc='count')
ts17_sci["Theme"] = "Origin"
ts12_sci = ts12_sci.append(ts17_sci)

ts18_sci = df_sci.pivot_table(index='Time', aggfunc='count')
ts18_sci["Theme"] = "All Themes"
ts12_sci = ts12_sci.append(ts18_sci)

theme_default = 'All Themes'
ts12_sci_filt = ts12_sci.query("Theme == '"+theme_default+"'")

ts12_sci_filt.drop(ts12_sci_filt.columns[1], axis=1, inplace=True)
dates_sci = ts12_sci_filt.index.tolist()
counts3_sci = ts12_sci_filt['Tweets'].tolist()
data_sci = dict(dates_sci=dates_sci, counts3_sci=counts3_sci)
data_sci = pd.DataFrame.from_dict(data_sci)
data_sci['dates_sci'] = pd.to_datetime(data_sci['dates_sci'])

p3_sci = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Scientific Sources Over Time")

p3_sci.line(
        x="dates_sci", y="counts3_sci",
        line_width=0.5, line_color="dodgerblue",
        legend_label = "Tweets",
        source=data_sci
        )

p3_sci.xaxis.axis_label = 'Date'
p3_sci.yaxis.axis_label = 'Tweets'

p3_sci.legend.location = "top_left"

def update_p3_sci(attrname, old, new):
    theme3_sci = widget3_sci.value
    ts12_sci_filt = ts12_sci.query("Theme == '"+theme3_sci+"'")
    ts12_sci_filt.drop(ts12_sci_filt.columns[1], axis=1, inplace=True)
    dates_sci = ts12_sci_filt.index.tolist()
    counts3_sci = ts12_sci_filt['Tweets'].tolist()
    data_sci = dict(dates_sci=dates_sci, counts3_sci=counts3_sci)
    data_sci = pd.DataFrame.from_dict(data_sci)
    data_sci['dates_sci'] = pd.to_datetime(data_sci['dates_sci'])
    p3_sci = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Volume of Tweets for Scientific Sources Over Time")
    p3_sci.line(
    x="dates_sci", y='counts3_sci',
    line_width=0.5, line_color="dodgerblue",
    legend_label = "Total Tweets",
    source=data_sci
        )
    p3_sci.xaxis.axis_label = 'Date'
    p3_sci.yaxis.axis_label = 'Number of Tweets'
    p3_sci.legend.location = "top_left"
    p_sci.children[1].children[0].children[1] = p3_sci
    
widget3_sci = Select(title="Themes for Times Series Chart", value=theme_default, options=['All Themes', 'Vaccines', 'Mandates', 'Alternative Treatments', 'Health Organizations','Conspiracies', 'Origin']
)
widget3_sci.on_change('value', update_p3_sci)

################### CREATING SENTIMENT ANALYSIS PIE CHART FOR SCIENTIFIC SOURCES a.k.a p4_sci #########################

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df_sci['Subjectivity'] = df_sci['Tweets'].apply(getTextSubjectivity)
df_sci['Polarity'] = df_sci['Tweets'].apply(getTextPolarity)

df_sci = df_sci.drop(df_sci[df_sci['Tweets'] == ''].index)

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

df_sci['Score'] = df_sci['Polarity'].apply(getTextAnalysis)

positive_sci = df_sci[df_sci['Score'] == 'Positive']
negative_sci = df_sci[df_sci['Score'] == 'Negative']
neutral_sci = df_sci[df_sci['Score'] == 'Neutral']
values_sci = int(negative_sci.shape[0]/(df_sci.shape[0])*100), int(neutral_sci.shape[0]/(df_sci.shape[0])*100), int(positive_sci.shape[0]/(df_sci.shape[0])*100)
labels_sci = df_sci.groupby('Score').count().index.values

pie_sci = dict(zip(labels_sci, values_sci))

data_sci = pd.Series(pie_sci).reset_index(name='values_sci').rename(columns={'index': 'labels_sci'})
data_sci['angle'] = data_sci['values_sci']/data_sci['values_sci'].sum() * 2*pi
data_sci['color'] = Category10[len(pie_sci)]

p4_sci = figure(height=650, title="Sentiment Analysis for Tweets from Scientific Sources",
           tools="hover", tooltips="@labels_sci: @values_sci%", x_range=(-0.5, 1.0))

p4_sci.wedge(x=0, y=0, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='labels_sci', source=data_sci)

p4_sci.axis.axis_label = None
p4_sci.axis.visible = False
p4_sci.grid.grid_line_color = None


################### CREATING TAB LAYOUT FOR SCIENTIFIC SOURCES a.k.a p_sci #########################

p_sci = column(
    row(
        column(widget1_sci, p1_sci),
        column(p4_sci)),
    row(
        column(widget3_sci, p3_sci))) 

tab1 = Panel(child=p_all,title="All Sources")
tab2 = Panel(child=p_frin,title="Fringe Sources")
tab3 = Panel(child=p_main,title="Mainstream Sources")
tab4 = Panel(child=p_pol,title="Political Sources")
tab5 = Panel(child=p_sci,title="Scientific Sources")
tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5])
curdoc().add_root(tabs)


curdoc().title = "Twitter Sentiment Analysis Dashboard"
