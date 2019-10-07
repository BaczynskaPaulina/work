from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import socketserver
import simplejson
import cgi
import urllib
import nltk
import numpy as np
import random
import string

import json
import pandas as pd
from copy import deepcopy
from urllib.parse import unquote
import base64
from base64 import decodestring
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


from sys import argv

import requests
from datetime import datetime

import ssl
import re
import sys
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords

global remove_punct_dict
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

global lemmer
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def levenshtein(a,b):
    n, m = len(a), len(b)
    if n > m:
        a,b = b,a
        n,m = m,n
    
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]



global q_sent_tokens
global q_word_tokens
global questions_list
global campaigns

def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def import_data():
    message=pd.read_excel("nlp_base.xlsx", names=["Question","Category"])
    message=message.applymap(str.lower)
    questions=list(message["Question"])
    categories=list(message["Category"])
    
    questions2=deepcopy(questions)
    for q in questions2:
        q=q.lower()
    
        questions_list=''
        for q in questions2:
            questions_list=questions_list+' '+q

    q_sent_tokens=nltk.sent_tokenize(questions_list)
    q_word_tokens=nltk.word_tokenize(questions_list)
    
    lemmer = nltk.stem.WordNetLemmatizer()
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    
    return questions, categories, q_sent_tokens, q_word_tokens, questions_list

questions, categories, q_sent_tokens, q_word_tokens, questions_list=import_data()

def get_listwords(questions_list):
    q_word_tokens2=nltk.word_tokenize(questions_list)
    for element in q_word_tokens2:
        if element=='.':
            q_word_tokens2.remove(element)
    for element in q_word_tokens2:
        if element=='(':
            q_word_tokens2.remove(element)
    for element in q_word_tokens2:
        if element==')':
            q_word_tokens2.remove(element)
    for element in q_word_tokens2:
        if element in stopwords.words('english'):
            q_word_tokens2.remove(element)
    q_word_tokens2=np.unique(q_word_tokens2)
    q_word_tokens2=list(q_word_tokens2)
    for element in q_word_tokens2:
        if len(element)<=2:
            q_word_tokens2.remove(element)
    for element in q_word_tokens2:
        if element==',':
            q_word_tokens2.remove(element)
    campaigns =list(q_word_tokens2)
    return campaigns

campaigns=get_listwords(questions_list)

def get_campaign(word):
    return min(campaigns, key=lambda x: levenshtein(word, x))


def find_answer(problem,questions, categories, q_sent_tokens, q_word_tokens):
    
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    user_response=deepcopy(problem)
    user_response=user_response.lower()
    
    q_sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_q = TfidfVec.fit_transform(q_sent_tokens)

    vals_q = cosine_similarity(tfidf_q[-1], tfidf_q)
    propos_q=[]
    iters=0

    flat_q = vals_q.flatten()
    flat_q.sort()
    flat_q= np.delete(flat_q, -1)
    idx_q=vals_q.argsort()
    idx_q= np.delete(idx_q, -1)
    idx_q=list(idx_q)


    for i in range(len((flat_q))):
        if (flat_q[i]>0.2):
            propos_q.append(idx_q[i])
            iters=+1
            
    sents=[]
    cats=[]
    req_tfidf = flat_q[-1]
    
    if (req_tfidf==0 or len(propos_q)==0):
        print("zero")
    
    else:
        if len(propos_q)!=0:
            for i in range(len(propos_q)):
                idxx=idx_q[-i-1]
                q_sent_tokens[idxx]=q_sent_tokens[idxx].lstrip()
                print(q_sent_tokens[idxx])
                print(idxx)
                if q_sent_tokens[idxx] in sents:
                    print("robo")
                else:
                    print("nie robo")
                    sents.append(q_sent_tokens[idxx])
                    cats.append(categories[idxx])
                
    cats_list=np.array(cats)
    cats_list=np.unique(cats_list)
    q_sent_tokens.remove(user_response)
    
    return cats_list




class Handler(BaseHTTPRequestHandler):
    
    
    def do_OPTIONS(self):
        print("OPTION")
        
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.send_header('Access-Control-Allow-Credentials','true')
        self.send_header('Access-Control-Allow-Headers','Access-Control-Allow-Origin,Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization,workspace-id')
        self.send_header('Access-Control-Allow-Methods','GET,PUT,POST,DELETE,PATCH,OPTIONS')
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Max-Age','1')
        
        
        self.end_headers()
        return
    
    def do_GET(self):
        print("GET")
        self.path = self.path+'ai/si_search/'
        self.send_response(200)
        self.end_headers()
        return


    def do_POST(self):
        print("POST")
        print(self.client_address)
        print("SELF: ",self.rfile)
        
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        print("POST BODY: ", post_body)
        
        d = json.loads(post_body)
        value=d['value']
        print("VALUE: ",value)

        
        value=value.rstrip()
        value.replace(u'\u2019', '-')
        value.encode('latin-1', 'replace')
        value.encode('UTF8', 'replace')
        value.encode('UTF8', 'replace')
        
        lista=[]
        lista=np.array(lista)
        
        if len(value)!=0:
            new_value=get_campaign(value)
            lista=find_answer(new_value,questions, categories, q_sent_tokens, q_word_tokens)
            print(lista)
            lista=list(lista)
            lista_str=json.dumps(lista)
            message=lista_str
            print(message)
            
            index=list(range(len(lista)))
            lista=np.delete(lista,index)
    
        else:
            message="{\"endingFlag\": false}"
            lista.delete()
            
        
        self.path = self.path+'ai/si_search/'
        request_path = self.path
        print(request_path)
        
        self.send_response(200)
        
        # Send headers
        self.send_header('Content-type','application/json')
        self.send_header('Access-Control-Allow-Credentials','true')
        self.send_header('Access-Control-Allow-Headers','Access-Control-Allow-Origin,Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization,workspace-id')
        self.send_header('Access-Control-Allow-Methods','GET,PUT,POST,DELETE,PATCH,OPTIONS')
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Max-Age','1')
        self.end_headers()

        thr =  threading.currentThread().getName()
        self.wfile.write(bytes(message, "utf8"))
        print("Thr: ",thr)
        print("Path: ",self.path)
        
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

def run():
    server = ThreadedHTTPServer(('192.168.1.5', 11080), Handler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()


run()
