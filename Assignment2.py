import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import string
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


consumer_key = "HmKPXaPlnJPDJiZRAp95GLGw3"
consumer_secret = "opHmJnzQQkxEFglzcb2ErPRNVMregWjAnxAjviI0yaHcuwm2FU"
access_key = "2564713387-VOo9LogVCx4xIgoBmsqxbru5Lhr2wRTcyHImRsD"
access_secret = "vi7NjnS49ZcWywndQyLc91UDoBbZ1gB1FGE9q9Fgs62KG"

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

# words=["anus","arse", "arsehole", "ass", "ass-hat","ass-jabber","ass-pirate","assbag", "assbandit","assbanger","assbite", "assclown","asscock", "asscracker", "asses", "assface", "assfuck", "assfucker", "assgoblin","asshat", "asshead","asshole","asshopper", "assjacker", "asslick", "asslicker", "assmonkey","assmunch", "assmuncher", "assnigger","asspirate","assshit", "assshole","asssucker", "asswad", "asswipe", "axwound", "bampot","bastard","beaner","beartrap", "bitch", "bitchass", "bitches","bitchtits", "bitchy","blow job","blowjob","bollocks","bollox","boner","brotherfucker","bullshit","bumblefuck","butt plug","butt-pirate", "buttfucka","buttfucker","camel toe", "carpetmuncher", "chesticle","chinc", "chink","choad","chode", "clit", "clitface", "clitfuck", "clitweasel", "clusterfuck", "cock", "cockass", "cockbite", "cockburger", "cockface", "cockfucker", "cockhead", "cockjockey", "cockknoker", "cockmaster", "cockmongler", "cockmongruel","cockmonkey", "cockmuncher", "cocknose", "cocknugget", "cockshit", "cocksmith", "cocksmoke","cocksmoker","cocksniffer","cocksucker", "cockwaffle", "coochie","coochy", "coon", "cooter", "cracker","cum", "cumbubble", "cumdumpster", "cumguzzler", "cumjockey", "cumslut", "cumtart","cunnie","cunnilingus", "cunt", "cuntass", "cuntface","cunthole","cuntlicker", "cuntrag", "cuntslut","dago", "damn", "deggo","dick","dick-sneeze", "dickbag", "dickbeaters", "dickface", "dickfuck", "dickfucker", "dickhead", "dickhole", "dickjuice", "dickmilk", "dickmonger","dicks","dickslap", "dicksucker", "dicksucking", "dicktickler", "dickwad", "dickweasel","dickweed", "dickwod", "dike", "dildo", "dipshit", "docking", "doochbag", "dookie", "douche", "douche-fag", "douchebag", "douchewaffle", "dumass", "dumb ass", "dumbass", "dumbfuck", "dumbshit", "dumshit", "dyke", "fag", "fagbag", "fagfucker", "faggit", "faggot", "faggotcock", "fagnut", "fagtard", "fatass", "fellatio", "feltch", "flamer", "fuck", "fuckass", "fuckbag", "fuckboy", "fuckbrain","fuckbutt","fuckbutter","fucked", "fucker", "fuckersucker", "fuckface", "fuckhead","fuckhole", "fuckin", "fucking", "fucknose", "fucknut", "fucknutt","fuckoff","fucks","fuckstick", "fucktard", "fucktart", "fuckup", "fuckwad", "fuckwit","fuckwitt", "fudgepacker", "gay", "gayass", "gaybob", "gaydo", "gayfuck", "gayfuckist", "gaylord","gaytard","gaywad", "goddamn", "goddamnit", "gooch", "gook", "goopchute", "gringo", "guido", "handjob", "hard on", "heeb", "hell","ho", "hoe","homo", "homodumbshit", "honkey", "humping", "jackass", "jagoff", "jap", "jerk off","jerkass", "jigaboo", "jizz","jungle bunny","junglebunny", "kike", "kooch", "kootch", "kraut", "kunt", "kyke", "lameass","lardass","lesbian", "lesbo", "lezzie", "masturbate", "mcfagget", "mick","minge", "mothafucka", "mothafuckin\'", "motherfucker","motherfucking", "muff", "muffdiver","munging","nazi", "negro", "nigaboo", "nigga", "nigger","niggerish", "niggers", "niglet", "nignog","nut sack", "nutsack", "paki", "panooch", "pecker","peckerhead", "penis", "penisbanger","penisfucker", "penispuffer", "piss", "pissed", "pissed off", "pissflaps", "polesmoker","pollock", "poon","poonani", "poonany", "poontang", "porch monkey", "porchmonkey", "prick","punanny", "punta", "pussies", "pussy", "pussylicking", "puto","queef", "queer","queerbait", "queerhole", "renob", "rimjob", "ruski", "sand nigger", "sandnigger", "schlong", "scrote", "shit", "shitass", "shitbag", "shitbagger", "shitbrains", "shitbreath", "shitcanned", "shitcunt", "shitdick", "shitface", "shitfaced","shithead", "shithole", "shithouse", "shitspitter", "shitstain", "shitter", "shittiest", "shitting", "shitty", "shiz", "shiznit", "skank", "skeet", "skullfuck", "slut", "slutbag", "smeg", "snatch", "spic", "spick", "splooge", "spook","suckass", "tard", "testicle", "thundercunt", "tit", "titfuck", "tits", "tittyfuck", "twat", "twatlips", "twats", "twatwaffle", "twit","uglyfuck", "unclefucker","va-j-j", "vag", "vagina", "vajayjay","vjayjay", "wank", "wankjob", "wetback", "whore", "whorebag", "whoreface", "wop" ]
# games=["fortnite","minecraft","PUBG"]

# extractor = twitter_setup()

# def get_tweets(api, words, number_of_tweets):
#     tweets = []
#     for w in words:
#         new_keyword = w + " -filter:retweets"
#         for content in tweepy.Cursor(api.search_tweets, q=new_keyword, lang="en").items(number_of_tweets):
#             tweets.append(content)   
#     return tweets


# alltweets = get_tweets(extractor, words, 100)
# print("number of tweets extracted:", len(alltweets))

# print("5 recent tweets:\n")
# for tweet in alltweets[:5]:
#     print(tweet.text)
    
# data = pd.DataFrame(data=[tweet.text for tweet in alltweets], columns=['Tweets'])
# data['ID'] = [tweet.id for tweet in alltweets]
# data['Date'] = [tweet.created_at for tweet in alltweets]
# data['Source'] = [tweet.source for tweet in alltweets]
# data['Likes'] = [tweet.favorite_count for tweet in alltweets]
# data['RTs'] = [tweet.retweet_count for tweet in alltweets]
# data['Location'] = [tweet.user.location for tweet in alltweets]

# data.to_csv('Cyberbullying.csv')


# gametweets = get_tweets(extractor, games, 300)
# print("number of tweets extracted:", len(gametweets))

# print("5 recent tweets:\n")
# for tweet in gametweets[:5]:
#     print(tweet.text,"\n")

# gamedata = pd.DataFrame(data=[tweet.text for tweet in gametweets], columns=['Tweets'])
# gamedata['ID'] = [tweet.id for tweet in gametweets]
# gamedata['Date'] = [tweet.created_at for tweet in gametweets]
# gamedata['Source'] = [tweet.source for tweet in gametweets]
# gamedata['Likes'] = [tweet.favorite_count for tweet in gametweets]
# gamedata['RTs'] = [tweet.retweet_count for tweet in gametweets]
# gamedata['Location'] = [tweet.user.location for tweet in gametweets]

# gamedata.to_csv('GameTweets.csv')

bullytweets = pd.read_csv('Cyberbullying.csv')
testtweets = pd.read_csv('GameTweets.csv')

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos= 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words):
    cleaned_token = []
    for token in tweet_tokens:
        token = re.sub('http[s]','', token)
        token = re.sub('//t.co/[A-Za-z0-9]+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)
        token = re.sub('[0-9]','', token)
        if (len(token) > 3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_token.append(token.lower())
    return cleaned_token

stop_words = stopwords.words('english')
stop_words.extend(['cyberbully','game','minecraft','play','update','PUBG','fortnite','build','building'])

text_blob = []
for tweet in bullytweets['Tweets'].tolist():
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity == 0:
        sentiment = "Neutral"
    elif analysis.sentiment.polarity > 0:
        sentiment = "Positive"
    elif analysis.sentiment.polarity < 0:
        sentiment = "Negative"
    text_blob.append(sentiment)

bullytweets['Sentiment'] = text_blob

labelled_tweets = bullytweets[['Tweets', 'Sentiment']]
labelled_tweets.drop(labelled_tweets.loc[labelled_tweets['Sentiment'] == 'Neutral'].index, inplace=True)

bullytweets_token = labelled_tweets['Tweets'].apply(word_tokenize).tolist()

cleaned_tokens = []
for tokens in bullytweets_token:
    rm_noise = remove_noise(tokens, stop_words)
    lemma_tokens = lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)
    
new_bullytweet = []
for line in cleaned_tokens:
    line = ' '.join(line)
    new_bullytweet.append(line)
    
tf = TfidfVectorizer(max_features=1000)
X = tf.fit_transform(new_bullytweet).toarray()
columns = tf.get_feature_names_out()
df = pd.DataFrame(X, columns = columns)

y = labelled_tweets['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cf = classification_report(y_test, y_pred)
print(cf)

labelled_testtweets = testtweets['Tweets']
testtweets_token = labelled_testtweets.apply(word_tokenize).tolist()

cleaned_testtokens = []
for tokens in testtweets_token:
    rm_noise = remove_noise(tokens, stop_words)
    lemma_tokens = lemmatize_sentence(rm_noise)
    cleaned_testtokens.append(lemma_tokens)
    
new_testtweet = []
for line in cleaned_testtokens:
    line = ' '.join(line)
    new_testtweet.append(line)
    
tf = TfidfVectorizer(max_features=1000)
T = tf.fit_transform(new_testtweet).toarray()
T_pred = model.predict(T)

T_pred = np.where(T_pred == "Positive", "No", T_pred)
T_pred = np.where(T_pred == "Negative", "Yes", T_pred)


harmfulness_pred = pd.DataFrame(T_pred,columns=["Harmfulness"])

output = labelled_testtweets.to_frame().merge(harmfulness_pred, left_index=True, right_index=True)

output.to_csv('Detection.csv')