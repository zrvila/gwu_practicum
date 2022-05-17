
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
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

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

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 5000000

########### Creating a SQL connection to our SQLite database ###########
con = sqlite3.connect("gettr.sqlite")

df= pd.read_sql('select * from gettr_table', con)
df_kmeans= pd.read_sql('select * from gettr_table', con)

########### Creating dataframes for each Theme ###########
Vaccines=df.post.str.contains("vaccines?|vax+|jabs?|mrna|biontech|pfizer|moderna|J&J|Johnson\s?&\s?Johnson", flags=re.IGNORECASE)
Vaccines =df.loc[Vaccines]

Mandates=df.post.str.contains("mandates?|mask mandates?|vaccine mandates?|vaccine cards?|passports?|lockdowns?|quarantines?|restrictions?", flags=re.IGNORECASE)
Mandates =df.loc[Mandates]

Alternative=df.post.str.contains("vitamins?|zinc|ivermectin", flags=re.IGNORECASE)
Alternative =df.loc[Alternative]

Health=df.post.str.contains("CDC|NIH|FDA|Centers for Disease Control|National Institutes of Health|Food and Drug Administration|World Health Organization", flags=re.IGNORECASE)|df.post.str.contains("WHO")
Health =df.loc[Health]

Consipiracies=df.post.str.contains("bioweapons?|labs?", flags=re.IGNORECASE)
Consipiracies =df.loc[Consipiracies]

Origin =df.post.str.contains("Wuhan|China", flags=re.IGNORECASE)
Origin =df.loc[Origin]

#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

#creating distinct dfs
all_sentences_vac= [sentence for sentence in Vaccines.post]
all_sentences_man = [sentence for sentence in Mandates.post]
all_sentences_alt = [sentence for sentence in Alternative.post]
all_sentences_hea = [sentence for sentence in Health.post]
all_sentences_con = [sentence for sentence in Consipiracies.post]
all_sentences_ori = [sentence for sentence in Origin.post]
all_sentences = [sentence for sentence in df.post]


############ 1a. Creating Vaccine Word Count dataframe ############
lines = []
for sentence in all_sentences_vac:
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
df12 = pd.DataFrame(stem2, columns=['Word'])
df12= df12['Word'].value_counts()
df12 = pd.DataFrame(df12)
df12['Theme'] = "Vaccines"

############ 1b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_vac = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_vac = dfp_vac.loc[(dfp_vac['Entity'] == 'PERSON')]
dfp_vac = dfp_vac['Word'].value_counts()
dfp = pd.DataFrame(dfp_vac)
dfp['Theme'] = "Vaccines"

############ 2a. Creating Mandates Word Count dataframe ############
lines = []
for sentence in all_sentences_man:
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
df13 = pd.DataFrame(stem2, columns=['Word'])
df13 = df13['Word'].value_counts()
df13 = pd.DataFrame(df13)
df13['Theme'] = "Mandates"
df12=df12.append(df13)

############ 2b. Creating Vaccine Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_man = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_man = dfp_man.loc[(dfp_man['Entity'] == 'PERSON')]
dfp_man = dfp_man['Word'].value_counts()
dfp_man = pd.DataFrame(dfp_man)
dfp_man['Theme'] = "Mandates"
dfp = dfp.append(dfp_man)

############ 3a. Creating Alternative Treatments Word Count dataframe ############
lines = []
for sentence in all_sentences_alt:
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
df14 = pd.DataFrame(stem2, columns=['Word'])
df14 = df14['Word'].value_counts()
df14 = pd.DataFrame(df14)
df14['Theme'] = "Alternative_Treatments"
df12=df12.append(df14)

############ 3b. Creating Alternative Treatments Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_alt = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_alt = dfp_alt.loc[(dfp_alt['Entity'] == 'PERSON')]
dfp_alt = dfp_alt['Word'].value_counts()
dfp_alt = pd.DataFrame(dfp_alt)
dfp_alt['Theme'] = "Alternative_Treatments"
dfp = dfp.append(dfp_alt)

############ 4a. Creating Health Organizations Word Count dataframe ############
lines = []
for sentence in all_sentences_hea:
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
df15 = pd.DataFrame(stem2, columns=['Word'])
df15 = df15['Word'].value_counts()
df15 = pd.DataFrame(df15)
df15['Theme'] = "Health_Organizations"
df12=df12.append(df15)

############ 4b. Creating Health Organizations Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_hea = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_hea = dfp_hea.loc[(dfp_hea['Entity'] == 'PERSON')]
dfp_hea = dfp_hea['Word'].value_counts()
dfp_hea = pd.DataFrame(dfp_hea)
dfp_hea['Theme'] = "Health_Organizations"
dfp = dfp.append(dfp_hea)

############ 5a. Creating Conspiracies Word Count dataframe ############
lines = []
for sentence in all_sentences_con:
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
df16 = pd.DataFrame(stem2, columns=['Word'])
df16 = df16['Word'].value_counts()
df16 = pd.DataFrame(df16)
df16['Theme'] = "Conspiracies"
df12 = df12.append(df16)

############ 5b. Creating Conspiracies Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_con = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_con = dfp_con.loc[(dfp_con['Entity'] == 'PERSON')]
dfp_con = dfp_con['Word'].value_counts()
dfp_con = pd.DataFrame(dfp_con)
dfp_con['Theme'] = "Conspiracies"
dfp = dfp.append(dfp_con)

############ 6a. Creating Origin Word Count dataframe ############
for sentence in all_sentences_ori:
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
df17 = pd.DataFrame(stem2, columns=['Word'])
df17 = df17['Word'].value_counts()
df17 = pd.DataFrame(df17)
df17['Theme'] = "Origin"
df12=df12.append(df17)

############ 6b. Creating Origin Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_ori = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_ori = dfp_ori.loc[(dfp_ori['Entity'] == 'PERSON')]
dfp_ori = dfp_ori['Word'].value_counts()
dfp_ori = pd.DataFrame(dfp_ori)
dfp_ori['Theme'] = "Origin"
dfp = dfp.append(dfp_ori)

############ 7a. Creating All Themes Word Count dataframe ############
lines = []
for sentence in all_sentences:
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
df18 = pd.DataFrame(stem2, columns=['Word'])
df18 = df18['Word'].value_counts()
df18 = pd.DataFrame(df18)
df18['Theme'] = "All"
df12=df12.append(df18)

############ 7b. Creating All Themes Person Count dataframe ############
str1 = " "
stem2 = str1.join(lines2)
stem2 = nlp(stem2)
label = [(X.text, X.label_) for X in stem2.ents]
dfp_all = pd.DataFrame(label, columns=['Word', 'Entity'])
dfp_all = dfp_all.loc[(dfp_all['Entity'] == 'PERSON')]
dfp_all = dfp_all['Word'].value_counts()
dfp_all = pd.DataFrame(dfp_all)
dfp_all['Theme'] = "All"
dfp = dfp.append(dfp_all)

#bokeh ----------------word-------------- cnts plotted

theme_default = 'All'
df12_filt = df12.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
words = df12_filt.index.tolist()
counts = df12_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source1 = ColumnDataSource(
    data=dict(words=words, counts=counts, color=pally))

# create a new plot with a title and axis labels
p1 = figure(y_range=words,
            x_range=(0, max(counts)),
            title='Top 20 Gettr Word Frequencies',
            y_axis_label='Words in Posts',
            x_axis_label='Word Count',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p1.hbar(y='words',
        right='counts',
        height=0.2,
        color='color',
        source=word_source1)
def update_plot(attrname, old, new):
    theme = widget1.value
    df12_filt = df12.query("Theme =='"+theme+"'").sort_values('Word',ascending=False).head(20)
    words = df12_filt.index.tolist()
    counts = df12_filt['Word'].tolist()
    word_source1.data=dict(words=words, counts=counts, color=pally)
    p1.y_range.factors=words

widget1 = Select(title="Themes for Word Bar Chart", value=theme_default, options=['All', 'Vaccines', 'Mandates', 'Alternative_Treatments', 'Health_Organizations','Conspiracies', 'Origin'])
widget1.on_change('value', update_plot)

#bokeh ----------------person-------------- cnts plotted

theme_default = 'All'
dfp_filt = dfp.query("Theme == '"+theme_default+"'").sort_values('Word', ascending=False).head(20)

# prepare data
people = dfp_filt.index.tolist()
counts2 = dfp_filt['Word'].tolist()

# generate color palatte for 20 colors
pally = all_palettes['Category20b'][20]

word_source2 = ColumnDataSource(
    data=dict(people=people, counts2=counts2, color=pally))

# create a new plot with a title and axis labels
p2 = figure(y_range=people,
            x_range=(0, max(counts2)),
            title='Top 20 Gettr Person References',
            y_axis_label='Person',
            x_axis_label='References',
            width=800,
            height=600)

# add a line renderer with legend and line thickness
p2.hbar(y='people',
        right='counts2',
        height=0.2,
        color='color',
        source=word_source2)
def update_plot2(attrname, old, new):
    theme2 = widget2.value
    dfp_filt = dfp.query("Theme =='"+theme2+"'").sort_values('Word',ascending=False).head(20)
    people = dfp_filt.index.tolist()
    counts2 = dfp_filt['Word'].tolist()
    word_source2.data=dict(people=people, counts2=counts2, color=pally)
    p2.y_range.factors=people

widget2 = Select(title="Themes for Person Bar Chart", value=theme_default, options=['All', 'Vaccines', 'Mandates', 'Alternative_Treatments', 'Health_Organizations','Conspiracies', 'Origin'])
widget2.on_change('value', update_plot2)

#bokeh ----------------Times Series -------------- cnts plotted
ts = df

# prepare ts data
ts_all = ts.pivot_table(index='date', aggfunc='count')
ts_all["Theme"] = "All"

ts_Vaccines = Vaccines.pivot_table(index='date', aggfunc='count')
ts_Vaccines["Theme"] = "Vaccines"
ts_all = ts_all.append(ts_Vaccines)

ts_Mandates = Mandates.pivot_table(index='date', aggfunc='count')
ts_Mandates["Theme"] = "Mandates"
ts_all = ts_all.append(ts_Mandates)

ts_Alternative = Alternative.pivot_table(index='date', aggfunc='count')
ts_Alternative["Theme"] = "Alternative_Treatments"
ts_all = ts_all.append(ts_Alternative)

ts_Health = Health.pivot_table(index='date', aggfunc='count')
ts_Health["Theme"] = "Health_Organizations"
ts_all = ts_all.append(ts_Health)

ts_Consipiracies = Consipiracies.pivot_table(index='date', aggfunc='count')
ts_Consipiracies["Theme"] = "Conspiracies"
ts_all = ts_all.append(ts_Consipiracies)

ts_Origin = Origin.pivot_table(index='date', aggfunc='count')
ts_Origin["Theme"] = "Origin"
ts_all = ts_all.append(ts_Origin)

theme_default = 'All'
ts_all_filt = ts_all.query("Theme == '"+theme_default+"'")

ts_all_filt.drop(ts_all_filt.columns[1], axis=1, inplace=True)
dates = ts_all_filt.index.tolist()
counts3 = ts_all_filt['post'].tolist()
data = dict(dates=dates, counts3=counts3)
data = pd.DataFrame.from_dict(data)
data['dates'] = pd.to_datetime(data['dates'])

line_chart = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Gettr Posts Times Series")

line_chart.line(
        x="dates", y="counts3",
        line_width=0.5, line_color="dodgerblue",
        legend_label = "Posts",
        source=data
        )

line_chart.xaxis.axis_label = 'Date'
line_chart.yaxis.axis_label = 'Posts'

line_chart.legend.location = "top_left"

def update_line_chart(attrname, old, new):
    theme3 = widget3.value
    ts_all_filt = ts_all.query("Theme == '"+theme3+"'")
    ts_all_filt.drop(ts_all_filt.columns[1], axis=1, inplace=True)
    dates = ts_all_filt.index.tolist()
    counts3 = ts_all_filt['post'].tolist()
    data = dict(dates=dates, counts3=counts3)
    data = pd.DataFrame.from_dict(data)
    data['dates'] = pd.to_datetime(data['dates'])
    line_chart = figure(plot_width=1600, plot_height=600, x_axis_type="datetime",
                    title="Gettr Posts Times Series")
    line_chart.line(
    x="dates", y='counts3',
    line_width=0.5, line_color="dodgerblue",
    legend_label = "Posts",
    source=data
        )
    line_chart.xaxis.axis_label = 'Date'
    line_chart.yaxis.axis_label = 'Posts'
    line_chart.legend.location = "top_left"
    p.children[1].children[0].children[1] = line_chart
    
widget3 = Select(title="Themes for Person Bar Chart", value=theme_default, options=['All', 'Vaccines', 'Mandates', 'Alternative_Treatments', 'Health_Organizations','Conspiracies', 'Origin']
)
widget3.on_change('value', update_line_chart)

#Bokeh ---------------- Sentiment Analysis -------------- Pie Chart 

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df['Subjectivity'] = df['post'].apply(getTextSubjectivity)
df['Polarity'] = df['post'].apply(getTextPolarity)

df = df.drop(df[df['post'] == ''].index)

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

df['Score'] = df['Polarity'].apply(getTextAnalysis)

positive = df[df['Score'] == 'Positive']
negative = df[df['Score'] == 'Negative']
neutral = df[df['Score'] == 'Neutral']
values = int(negative.shape[0]/(df.shape[0])*100), int(neutral.shape[0]/(df.shape[0])*100), int(positive.shape[0]/(df.shape[0])*100)
labels = df.groupby('Score').count().index.values

pie = dict(zip(labels, values))

data = pd.Series(pie).reset_index(name='values').rename(columns={'index': 'labels'})
data['angle'] = data['values']/data['values'].sum() * 2*pi
data['color'] = Category10[len(pie)]

p4 = figure(height=650, title="Sentiment Analysis for All Posts",
           tools="hover", tooltips="@labels: @values%", x_range=(-0.5, 1.0))

p4.wedge(x=0, y=0, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='labels', source=data)

p4.axis.axis_label = None
p4.axis.visible = False
p4.grid.grid_line_color = None

#bokeh ----------------KMEANS Scatter Plot -------------- ##############

#lowercase all text
df_kmeans['post'] = df_kmeans['post'].str.lower()

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
punctuation = string.punctuation.split(
)  #already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()


def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)


df_kmeans.post = df_kmeans.post.apply(furnished)

#save processed text to objects
mainstream = furnished(mainstream_related_words)
fringe = furnished(fringe_related_words)

#clean text for visual
string1 = mainstream
words = string1.split()
mainstream = " ".join(sorted(set(words), key=words.index))
mainstream
string1 = fringe
words = string1.split()
fringe = " ".join(sorted(set(words), key=words.index))
fringe


#define scoring functions for text
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def get_scores(group, posts):
    scores = []
    for post in posts:
        s = jaccard_similarity(group, post)
        scores.append(s)
    return scores


m_scores = get_scores(mainstream, df.post.to_list())
f_scores = get_scores(fringe, df.post.to_list())

# create a jaccard scored df.
data  = {'names':df.username.to_list(),'mainstream_score':m_scores, 'fringe_score': f_scores}
scores_df = pd.DataFrame(data)
#assign classes based on highest score
def get_classes(l1, l2):
    main = []
    frin = []
    for i, j, in zip(l1, l2):
        m = max(i, j)
        if m == i:
            main.append(1)
        else:
            main.append(0)
        if m == j:
            frin.append(1)
        else:
            frin.append(0)        
            
    return main, frin
l1 = scores_df.mainstream_score.to_list()
l2 = scores_df.fringe_score.to_list()

main, frin = get_classes(l1, l2)
data = {'username': scores_df.names.to_list(), 'mainstream':main, 'fringe':frin}
class_df = pd.DataFrame(data)

#grouping the tweets by username
new_groups_df = class_df.groupby(['username']).sum()

# fitting kmeans to dataset
X = new_groups_df[['mainstream', 'fringe']].values

kmeans = KMeans(n_clusters=3)
kmeans.fit_predict(X)

# Bokeh plot
# Create a blank figure with necessary arguments
p5 = figure(plot_width=800, plot_height=650,title='KMeans Post Clustering: Mainstream & Fringe')
p5.xaxis.axis_label = 'Mainstream Post Count'
p5.yaxis.axis_label = 'Fringe Post Count'

clus_xs = []

clus_ys = []

for entry in kmeans.cluster_centers_:
    clus_xs.append(entry[0])
    clus_ys.append(entry[1])
    
p5.circle_cross(x=clus_xs, y=clus_ys, size=40, fill_alpha=0, line_width=2, color=['pink', 'red', 'purple'])
p5.text(text = ['Cluster 1', 'Cluster 2', 'Cluster 3'], x=clus_xs, y=clus_ys, text_font_size='30pt')
p5.x_range=Range1d(0,400)
p5.y_range=Range1d(0,400)


i = 0 #counter

for sample in X:
    if kmeans.labels_[i] == 0:
        p5.circle(x=sample[0], y=sample[1], size=15, color="pink")
    if kmeans.labels_[i] == 1:
        p5.circle(x=sample[0], y=sample[1], size=15, color="red")
    if kmeans.labels_[i] == 2:
        p5.circle(x=sample[0], y=sample[1], size=15, color="purple")
    i += 1

p = column(
    row(
        column(widget1, p1),
        column(widget2, p2),
        column(p4)),
    row(
        column(widget3, line_chart),
        column(p5))) 
    
#l1 = column(
    #row(
        #column(widget1, p1),
        #column(widget2, p2)))
#l2 = column(
    #row(
        #column(widget3, p3),
        #column(p5)))


#tab1 = Panel(child=l1,title="This is Tab 1")
#tab2 = Panel(child=l2,title="This is Tab 2")
#tabs = Tabs(tabs=[ tab1, tab2 ])

#curdoc().add_root(tabs)

#print results to webpage
output_file('COVID-19 Gettr Dashboard.html')

curdoc().add_root(p)
curdoc().title = "Gettr Sentiment Analysis Dashboard"

##### Melis link:###### bokeh serve app.py --allow-websocket-origin=________________________:5006 --show
## bokeh serve myapp/ --allow-websocket-origin=ec2-18-207-115-152.compute-1.amazonaws.com:5006 --show