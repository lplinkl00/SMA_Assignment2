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

words=["anus","arse", "arsehole", "ass", "ass-hat","ass-jabber","ass-pirate","assbag", "assbandit","assbanger","assbite", "assclown","asscock", "asscracker", "asses", "assface", "assfuck", "assfucker", "assgoblin","asshat", "asshead","asshole","asshopper", "assjacker", "asslick", "asslicker", "assmonkey","assmunch", "assmuncher", "assnigger","asspirate","assshit", "assshole","asssucker", "asswad", "asswipe", "axwound", "bampot","bastard","beaner","beartrap", "bitch", "bitchass", "bitches","bitchtits", "bitchy","blow job","blowjob","bollocks","bollox","boner","brotherfucker","bullshit","bumblefuck","butt plug","butt-pirate", "buttfucka","buttfucker","camel toe", "carpetmuncher", "chesticle","chinc", "chink","choad","chode", "clit", "clitface", "clitfuck", "clitweasel", "clusterfuck", "cock", "cockass", "cockbite", "cockburger", "cockface", "cockfucker", "cockhead", "cockjockey", "cockknoker", "cockmaster", "cockmongler", "cockmongruel","cockmonkey", "cockmuncher", "cocknose", "cocknugget", "cockshit", "cocksmith", "cocksmoke","cocksmoker","cocksniffer","cocksucker", "cockwaffle", "coochie","coochy", "coon", "cooter", "cracker","cum", "cumbubble", "cumdumpster", "cumguzzler", "cumjockey", "cumslut", "cumtart","cunnie","cunnilingus", "cunt", "cuntass", "cuntface","cunthole","cuntlicker", "cuntrag", "cuntslut","dago", "damn", "deggo","dick","dick-sneeze", "dickbag", "dickbeaters", "dickface", "dickfuck", "dickfucker", "dickhead", "dickhole", "dickjuice", "dickmilk", "dickmonger","dicks","dickslap", "dicksucker", "dicksucking", "dicktickler", "dickwad", "dickweasel","dickweed", "dickwod", "dike", "dildo", "dipshit", "docking", "doochbag", "dookie", "douche", "douche-fag", "douchebag", "douchewaffle", "dumass", "dumb ass", "dumbass", "dumbfuck", "dumbshit", "dumshit", "dyke", "fag", "fagbag", "fagfucker", "faggit", "faggot", "faggotcock", "fagnut", "fagtard", "fatass", "fellatio", "feltch", "flamer", "fuck", "fuckass", "fuckbag", "fuckboy", "fuckbrain","fuckbutt","fuckbutter","fucked", "fucker", "fuckersucker", "fuckface", "fuckhead","fuckhole", "fuckin", "fucking", "fucknose", "fucknut", "fucknutt","fuckoff","fucks","fuckstick", "fucktard", "fucktart", "fuckup", "fuckwad", "fuckwit","fuckwitt", "fudgepacker", "gay", "gayass", "gaybob", "gaydo", "gayfuck", "gayfuckist", "gaylord","gaytard","gaywad", "goddamn", "goddamnit", "gooch", "gook", "goopchute", "gringo", "guido", "handjob", "hard on", "heeb", "hell","ho", "hoe","homo", "homodumbshit", "honkey", "humping", "jackass", "jagoff", "jap", "jerk off","jerkass", "jigaboo", "jizz","jungle bunny","junglebunny", "kike", "kooch", "kootch", "kraut", "kunt", "kyke", "lameass","lardass","lesbian", "lesbo", "lezzie", "masturbate", "mcfagget", "mick","minge", "mothafucka", "mothafuckin\'", "motherfucker","motherfucking", "muff", "muffdiver","munging","nazi", "negro", "nigaboo", "nigga", "nigger","niggerish", "niggers", "niglet", "nignog","nut sack", "nutsack", "paki", "panooch", "pecker","peckerhead", "penis", "penisbanger","penisfucker", "penispuffer", "piss", "pissed", "pissed off", "pissflaps", "polesmoker","pollock", "poon","poonani", "poonany", "poontang", "porch monkey", "porchmonkey", "prick","punanny", "punta", "pussies", "pussy", "pussylicking", "puto","queef", "queer","queerbait", "queerhole", "renob", "rimjob", "ruski", "sand nigger", "sandnigger", "schlong", "scrote", "shit", "shitass", "shitbag", "shitbagger", "shitbrains", "shitbreath", "shitcanned", "shitcunt", "shitdick", "shitface", "shitfaced","shithead", "shithole", "shithouse", "shitspitter", "shitstain", "shitter", "shittiest", "shitting", "shitty", "shiz", "shiznit", "skank", "skeet", "skullfuck", "slut", "slutbag", "smeg", "snatch", "spic", "spick", "splooge", "spook","suckass", "tard", "testicle", "thundercunt", "tit", "titfuck", "tits", "tittyfuck", "twat", "twatlips", "twats", "twatwaffle", "twit","uglyfuck", "unclefucker","va-j-j", "vag", "vagina", "vajayjay","vjayjay", "wank", "wankjob", "wetback", "whore", "whorebag", "whoreface", "wop" ]

extractor = twitter_setup()
def get_tweets(api, words):
    tweets = []
    for w in words:
        for content in tweepy.Cursor(api.search_tweets, q=w, lang="en").items(100): #screen_name is the twitter handle, not the display name i.e the one with "@SomeUser"
            tweets.append(content)
    
    return tweets

alltweets = get_tweets(extractor, words)
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
