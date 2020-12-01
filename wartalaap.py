from collections import Counter
import calendar
import datetime
import emoji
import math
import os
import re
import regex
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import time
from matplotlib.colors import ColorConverter, ListedColormap
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

st.title('WArtalaap: Analyze your WhatsApp Chats')
st.markdown('Analyze your personal chats or group chats with WArtalaap!')
logo = Image.open("./logo.jpg")
st.sidebar.title("WArtalaap")
st.sidebar.image(logo, use_column_width=True)
st.sidebar.markdown("This app analyzes your WhatsApp Chats")

st.sidebar.markdown('[![Arghyadeep Das]\
					(https://img.shields.io/badge/Author-@arghyadeep99-gray.svg?colorA=gray&colorB=dodgerblue&logo=github)]\
					(https://github.com/arghyadeep99)')

st.sidebar.markdown('**How to export your chat in .txt format? (Not available on WhatsApp Web)**')
st.sidebar.markdown('1) Open your individual or group chat.')
st.sidebar.markdown('2) Press on the three dots on top right corner.')
st.sidebar.markdown('3) Tap More -> Export Chat.')
st.sidebar.markdown('4) When prompted, choose "Without Media" option as we are performing text analysis only.')
st.sidebar.markdown('5) Upload this exported .txt file to our app.')

st.sidebar.markdown('*You are all set to go!* 😃')
st.sidebar.subheader('**FAQs**')
st.sidebar.markdown('**Is my chat history saved on your servers?**')
st.sidebar.markdown('No, the data that you upload is not saved anywhere on this website or any 3rd party website i.e, not in any storage like database/File System/Logs/Cloud.')


def isAuthor(s):
	s=s.split(":")
	if len(s) >= 2:
		return True
	else:
		return False

def emoji_count(text):

	emoji_list = []
	data = regex.findall(r'\X', text)
	for word in data:
		if any(char in emoji.UNICODE_EMOJI for char in word):
			emoji_list.append(word)

	return emoji_list

def emoji_plot(data):
	total_emojis_list = list([a for b in data.emoji for a in b])
	emoji_dict = dict(Counter(total_emojis_list))
	emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
	emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])

	fig = px.pie(emoji_df, values='count', names='emoji')
	fig.update_traces(textposition='inside', textinfo='percent+value+label')
	fig.update_layout(
		margin=dict(l=5,r=5,)
		)
	fig.update(layout_showlegend=True)
	fig.update_layout(
		margin=dict(l=5,r=5,)
		)
	return fig

def daily_freq_plot(data):
	date_df = data.groupby(data['Date']).sum()
	date_df.reset_index(inplace=True)
	fig = px.line(date_df, x="Date", y="MessageCount", title='Daily Message Frequency Plot')
	fig.update_xaxes(nticks=20)
	return fig


def most_active_days(data):
	
	data['Date'].value_counts().head(10).plot.barh()
	plt.xlabel('Number of Messages')
	plt.ylabel('Date')
	plt.tight_layout()
	'''
	temp = pd.DataFrame(['Date','Count'])
	temp['Date'] = data['Date'].value_counts().head(10).index
	temp['Count'] = data['Date'].value_counts().head(10)
	fig = px.bar(temp, x='Count', y='Date', text='Count')
	fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
	fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
	fig.show()
	'''

def most_active_time(data):
	data['Time'].value_counts().head(10).plot.barh()
	plt.xlabel('Number of Messages')
	plt.ylabel('Date')
	plt.tight_layout()

def wordcloud_author(data):
	text = " ".join(review for review in data.Message)
	stop_words = set(STOPWORDS)
	stop_words.update(["kya", "Haa", "ko", "ye", "Ye", "se", "hai", "thi", "ka", "ha", "na", "toh", "hi", "mein", "no", "yes", "nahi", "koi", "ki"])
	wordcloud = WordCloud(stopwords=stop_words, background_color="white",height=640, width=800).generate(text)
	# Display the generated image the matplotlib way:
	plt.axis("off")
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	
def daywise_messages(data):
	def dayofweek(i):
		l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
		return l[i]

	day_df=pd.DataFrame(data["Message"])
	day_df['day_of_date'] = data['Date'].dt.weekday
	day_df['day_of_date'] = day_df["day_of_date"].apply(dayofweek)
	day_df["MessageCount"] = 1
	day = day_df.groupby("day_of_date").sum()
	day.reset_index(inplace=True)

	fig = px.line_polar(day, r='MessageCount', theta='day_of_date', line_close=True)
	fig.update_traces(fill='toself')
	fig.update_layout(
	  polar=dict(
		radialaxis=dict(
		  visible=True,
		  range=[0,int(math.ceil(day['MessageCount'].max().item()) + 100)]
		)),
	  showlegend=True,
	  margin=dict(l=5,r=5,)
	)
	return fig

def calendar_plot(data, year=None, how='count', column = 'MessageCount'):
	""" Adjusted calendar plot from https://pythonhosted.org/calmap/
	
	Copyright (c) 2015 by Martijn Vermaat
	
	
	To do:
	* year set to None and find the minimum year
	* Choose column instead of using index
	* Set date as index
	
	Parameters:
	-----------
	year : int, default None
	how : string, default 'count'
		Which methods to group by the values. 
		Note, it is always by day due to the
		nature of a calendar plot. 
	column : string, default 'User'
		The column over which you either count or sum the values
		For example, count activity in a single day.
		
	"""
	
	# Get minimum year if not given
	if year == None:
		year = data.Date.min().year
	
	# Prepare data
	data = data.set_index('Date').loc[:, column]
	
	# Resample data
	if how == 'sum':
		daily = data.resample('D').sum()
	elif how == 'count':
		daily = data.resample('D').count()
	
	vmin = daily.min()
	vmax = daily.max()

	# Fill in missing dates
	daily = daily.reindex(pd.date_range(start=str(year), end=str(year + 1), 
										freq='D')[:-1])
	daily = daily.to_frame()
	# Put into dataframe
	# Fill is needed to created the initial raster
	daily['fill'] = 1
	daily['day'] = daily.index.weekday
	daily['week'] = daily.index.week
	

	# Correctly choose week and day
	daily.loc[(daily.index.month == 1) & (daily.week > 50), 'week'] = 0
	daily.loc[(daily.index.month == 12) & (daily.week < 10), 'week'] \
		= daily.week.max() + 1

	# Create data to be plotted
	plot_data = daily.pivot('day', 'week', 'MessageCount').values[::-1]
	plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

	# Create data for the background (all days)
	fill_data = daily.pivot('day', 'week', 'fill').values[::-1]
	fill_data = np.ma.masked_where(np.isnan(fill_data), fill_data)

	# Set plotting values
	cmap='OrRd'
	linewidth=1
	linecolor = 'white'
	fillcolor='whitesmoke'

	# Draw heatmap for all days of the year with fill color.
	fig = plt.figure(figsize=(20, 10))
	ax = plt.gca()
	ax.pcolormesh(fill_data, vmin=0, vmax=1, cmap=ListedColormap([fillcolor]))
	ax.pcolormesh(plot_data, vmin=vmin, vmax=vmax, cmap=cmap, 
				  linewidth=linewidth, edgecolors=linecolor)

	# Limit heatmap to our data.
	ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))

	# # Square cells.
	ax.set_aspect('equal')

	# Remove spines and ticks.
	for side in ('top', 'right', 'left', 'bottom'):
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(which='both', length=0)
	ax.yaxis.set_tick_params(which='both', length=0)

	# Get ticks and labels for days and months
	daylabels = calendar.day_abbr[:]
	dayticks = range(len(daylabels))

	monthlabels = calendar.month_abbr[1:]
	monthticks = range(len(monthlabels))

	# Create label and ticks for x axis
	font = {'fontname':'Comic Sans MS', 'fontsize':20}
	ax.set_xlabel('')
	ax.set_xticks([3+i*4.3 for i in monthticks])
	ax.set_xticklabels([monthlabels[i] for i in monthticks], ha='center', **font)

	# Create label and ticks for y axis
	font = {'fontname':'Comic Sans MS', 'fontsize':15}
	ax.set_ylabel('')
	ax.yaxis.set_ticks_position('right')
	ax.set_yticks([6 - i + 0.5 for i in dayticks])
	ax.set_yticklabels([daylabels[i] for i in dayticks], rotation='horizontal',
					   va='center', **font)
	

	ax.set_ylabel(str(year), fontsize=52,color='#DCDCDC',fontweight='bold',
				  fontname='Comic Sans MS', ha='center')
	plt.tight_layout()
	plt.show()

uploaded_file = st.file_uploader("Upload Your Whatsapp Chat.(.txt file only!)", type="txt")
if uploaded_file is not None:
	@st.cache(allow_output_mutation=True)
	def import_data(file):
		next(file)
		df = pd.DataFrame(columns = ["Date", "Time", "Author", "Message"])
		lists = [[] for _ in range(4)]

		senders = []
		try:
			for message in file:
					if re.search(r"[\d]{1,2}/[\d]{1,2}/[\d]{2}, [\d]{1,2}:[\d]{1,2} ", message):
						date_time, sender_message = message.split("-")[0].strip(),' '.join(message.split("-")[1:])
						date, time = date_time[:8].strip(), date_time[date_time.index(",")+2:].strip()
						if isAuthor(sender_message):
							sender, chat_message = sender_message[:sender_message.index(":")].strip(), sender_message[sender_message.index(":")+2:].strip()
							if sender not in senders:
								senders.append(sender)
						else:
							sender, chat_message = None, sender_message.strip()

						lists[0].append(date)
						lists[1].append(time)
						lists[2].append(sender)
						lists[3].append(chat_message)
					else:
						lists[3][-1] += ' '+ message

		except Exception as e:
			print(e)

		for k, col in enumerate(df.columns):
			df[col] = lists[k]

		df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
		df = df.dropna()
		df["emoji"] = df.Message.apply(emoji_count)
		URLPATTERN = r'(https?://\S+)'
		df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
		return df

	df = import_data(uploaded_file)
	authors = list(df.Author.unique())
	authors.sort()
	authors.insert(0, 'All')
	st.subheader("Select the person whose stats you want to view:")
	option = st.selectbox("", authors)

	if option == 'All':
		name = "Group"
		link_messages= df[df['urlcount']>0]
		deleted_messages=df[(df["Message"] == "You deleted this message")| (df["Message"] == "This message was deleted")|(df["Message"] == "You deleted this message.")]
		media_messages_df = df[(df['Message'] == '<Media omitted>')|(df['Message'] == 'image omitted')|(df['Message'] == 'video omitted')|(df['Message'] == 'sticker omitted')]
		emojis = sum(df['emoji'].str.len())
		links = np.sum(df.urlcount)

		st.subheader("**%s's total messages 💬**"% name)
		st.markdown(df.shape[0])
		st.subheader("**%s's total media 🎬**"% name)
		st.markdown(media_messages_df.shape[0])
		st.subheader("**%s's total emojis **"% name)
		st.markdown(emojis)
		st.subheader("**%s's total links**"% name)
		st.markdown(links)
		messages_df = df.drop(media_messages_df.index)
		messages_df = messages_df.drop(deleted_messages.index)
		messages_df = messages_df.drop(link_messages.index)

		messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(re.findall(r'\w+', s)))
		messages_df["MessageCount"]=1
		messages_df["emojicount"]= messages_df['emoji'].str.len()
		config={'responsive': True}
		st.header("**Some more Stats**")
		st.subheader("**%s's emoji distribution**"% name)
		st.text("Hover on Chart to see details.")
		st.plotly_chart(emoji_plot(messages_df),use_container_width=True)
		st.subheader("**%s's message daily frequency plot**"% name)
		st.text("Hover on Chart to see details.")
		st.plotly_chart(daily_freq_plot(messages_df),use_container_width=True)
		st.subheader("**The most happening day for the %s was - **"% name)
		most_active_days(messages_df)
		st.pyplot()
		time.sleep(0.2)
		st.subheader("**When is the %s most active?**"% name)
		most_active_time(messages_df)
		st.pyplot()
		time.sleep(0.2)
		st.subheader("**%s's WordCloud **"% name)
		wordcloud_author(messages_df)
		st.pyplot()
		time.sleep(0.2)
		st.subheader(f"**{name}'s day-wise messages count**")
		st.text("Hover on Chart to see details.")
		st.plotly_chart(daywise_messages(messages_df),use_container_width=True)
		st.subheader(f"**{name}'s Calendar Plot**")
		years = set(pd.DatetimeIndex(messages_df.Date.values).year)
		for year in years:
			calendar_plot(messages_df, year=year, how='count', column='MessageCount')
			st.pyplot()
			time.sleep(0.2)
	
	else:
		df = df[df.Author.eq(option)]
		name = option
		link_messages= df[df['urlcount']>0]
		deleted_messages=df[(df["Message"] == "You deleted this message")| (df["Message"] == "This message was deleted")|(df["Message"] == "You deleted this message.")]
		media_messages_df = df[(df['Message'] == '<Media omitted>')|(df['Message'] == 'image omitted')|(df['Message'] == 'video omitted')|(df['Message'] == 'sticker omitted')]
		emojis = sum(df['emoji'].str.len())
		links = np.sum(df.urlcount)

		st.subheader("**%s's total messages 💬**"% name)
		st.markdown(df.shape[0])
		st.subheader("**%s's total media 🎬**"% name)
		st.markdown(media_messages_df.shape[0])
		st.subheader("**%s's total emojis **"% name)
		st.markdown(emojis)
		st.subheader("**%s's total links 🔗**"% name)
		st.markdown(links)
		messages_df = df.drop(media_messages_df.index)
		messages_df = messages_df.drop(deleted_messages.index)
		messages_df = messages_df.drop(link_messages.index)

		messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(re.findall(r'\w+', s)))
		messages_df["MessageCount"]=1
		messages_df["emojicount"]= messages_df['emoji'].str.len()
		st.subheader(f"**Average number of words per message 🔤**")
		st.markdown((np.sum(messages_df['Word_Count']))/messages_df.shape[0])
		config={'responsive': True}
		st.header("**Some more Stats**")
		st.subheader("**%s's emoji distribution**"% name)
		st.text("Hover on Chart to see details.")
		st.plotly_chart(emoji_plot(messages_df),use_container_width=True)
		st.subheader("**%s's message daily frequency plot**"% name)
		st.text("Hover on Chart to see details.")
		st.plotly_chart(daily_freq_plot(messages_df),use_container_width=True)
		st.subheader("**The top 10 happening day for %s were **"% name)
		most_active_days(messages_df)
		st.pyplot()
		time.sleep(0.2)
		st.subheader("**Top 10 times when %s is most active?**"% name)
		most_active_time(messages_df)
		st.pyplot()
		time.sleep(0.2)
		st.subheader("**%s's WordCloud **"% name)
		wordcloud_author(messages_df)
		st.pyplot()
		time.sleep(0.2)
		st.subheader(f"**{name}'s day-wise messages count**")
		st.text("Hover on Chart to see details.")
		st.plotly_chart(daywise_messages(messages_df),use_container_width=True)
		st.subheader(f"**{name}'s Calendar Plot**")
		years = set(pd.DatetimeIndex(messages_df.Date.values).year)
		for year in years:
			calendar_plot(messages_df, year=year, how='count', column='MessageCount')
			st.pyplot()
			time.sleep(0.2)
