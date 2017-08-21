import pandas as pd
import re
pd.set_option('display.max_colwidth', 200)

# Load the conversations downloaded from http://notsocleverbot.jimrule.com/ using import.io download service
df = pd.read_csv(r'nscb2.csv', encoding='latin-1')
df.head()

# Extract only the required info
convo = df.iloc[:,1]
#convo

# Use regular expression to from question response pairs
clist = []
def qa_pairs(x):
    cpairs = re.findall(": (.*?)(?:$|\\n)", x)
    clist.extend(list(zip(cpairs, cpairs[1:])))
convo.map(qa_pairs);
convo_frame = pd.Series(dict(clist)).to_frame().reset_index()
convo_frame.columns = ['q', 'a']
#convo_frame

# Transform the training data to Tf-IDF form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(ngram_range=(1,3))
vec = vectorizer.fit_transform(convo_frame['q'])

# Define a function to find the most matching response to an input question
def get_response(q):
    my_q = vectorizer.transform([q])
    cs = cosine_similarity(my_q, vec)
    rs = pd.Series(cs[0]).sort_values(ascending=0)
    rsi = rs.index[0]
    return convo_frame.iloc[rsi]['a']

# Testing the model
print(get_response('Yes, I am clearly more clever than you will ever be!'))

query = input("Enter a question!\n")
while(query != 'quit'):
	print(get_response(query))
	query = input("")
	
#get_response('You are a stupid machine. Why must I prove anything to you?')
#get_response('My spirit animal is a menacing cat. What is yours?')
#get_response('I mean I didn\'t actually name it.')
#get_response('Do you have a name suggestion?')
#get_response('I think it might be a bit aggressive for a kitten')
#get_response('No need to involve the police.')
#get_response('And I you, Cleverbot')
#get_response('Are you a Cleverbot?')
#get_response("Say goodbye, Clevercake")
#get_response("Say goodbye, Cleverbot")
#get_response("Goodbye?")