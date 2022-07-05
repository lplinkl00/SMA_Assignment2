import tweepy #install tweepy in the venv
import pandas as pd

consumer_key= "OnPcLNxSQBSgJ0uHPWRyJf3Vf"
consumer_secret= "NgOKHm27tZDnOLCt5NI6fUP0GcQ3hEgAMDBVadEvYqVElt4s3e"
access_key="2247116646-0sG9Qj6yhhM8bPoq7tDZsltq0j85j3eI6ZLwftp"
access_secret="uaqutud7jJA5SKl0iepraqSmsq091CSe4tGSuIsrysU3c"

def twitter_setup():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    try:
        api.verify_credentials()
        print("authentication OK")
    except:
        print("error during authentication")
    return api


extractor = twitter_setup()

def get_hashtag_tweets(api, hashtag):
    tweets = []
    for content in tweepy.Cursor(api.search_tweets, q=hashtag).items(): #screen_name is the twitter handle, not the display name i.e the one with "@SomeUser"
        tweets.append(content)
    
    return tweets

def keyword_tweets(api, keyword, number_of_tweets):
    new_keyword = keyword + " -filter:retweets"
    tweets = []
    for status in tweepy.Cursor(api.search_tweets, q=new_keyword, lang="en").items(number_of_tweets):
        tweets.append(status)
    
    return tweets

alltweets = get_hashtag_tweets(extractor, 'Cyberbullying')
print("number of tweets extracted:", len(alltweets))

print("5 recent tweets:\n")
for tweet in alltweets[:5]:
    print(tweet.text)
    
data = pd.DataFrame(data=[tweet.text for tweet in alltweets], columns=['Tweets'])
data['ID'] = [tweet.id for tweet in alltweets]
data['Date'] = [tweet.created_at for tweet in alltweets]
data['Source'] = [tweet.source for tweet in alltweets]
data['Likes'] = [tweet.favorite_count for tweet in alltweets]
data['RTs'] = [tweet.retweet_count for tweet in alltweets]
data['Location'] = [tweet.user.location for tweet in alltweets]

data.to_csv('Cyberbllying.csv')
