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

global threads #jest to lista wszystkich użytkowników bota
threads=[]

id=1 #id użytkownika, przy każdym starcie serwera numerujemy od 1

#ponizej wywołanie potrzebnych narzędzi NLP
global remove_punct_dict
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

global lemmer
lemmer = nltk.stem.WordNetLemmatizer()

global daytime
daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

global MAX_BUFFER_SIZE
MAX_BUFFER_SIZE = 4096

global q_sent_tokens
global q_word_tokens
global a_sent_tokens
global a_word_tokens

#poniższe trzy funkcje służą do obróbki wstępnej NLP

#lemmatyzacja
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

#tokenizacja
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#usunięcie stopword'ów
def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#poniższa funkcja służy do rozpoznania kategorii pytania
def question_classifier():
    message=pd.read_excel("dataset-ml-2.xlsx", names=["Question","Category"])
    message['Question'].head(5).apply(text_process)
    
    bow_transformer = CountVectorizer(analyzer=text_process).fit(message['Question'])
    messages_bow = bow_transformer.transform(message['Question'])
    sparsity =(100.0 * messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
    
    tfidf_transformer=TfidfTransformer().fit(messages_bow)
    messages_tfidf=tfidf_transformer.transform(messages_bow)
    category_detect_model = RidgeClassifierCV().fit(messages_tfidf,message['Category'])
    all_predictions = category_detect_model.predict(messages_tfidf)

    msg_train,msg_test,label_train,label_test = train_test_split(message['Question'],message['Category'],test_size=0.3)
    pipeline = Pipeline([( 'bow',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),('classifier',RidgeClassifierCV()),])
    pipeline.fit(msg_train,label_train)

    return pipeline

#tworzy nowy ticket w BaseCampie
def make_ticket(name,email,sentence,todolist):
    daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title_from_string=deepcopy(sentence)
    title=daytime+" | "+title_from_string+" | "+name+" | "+email
    
    url = "https://basecamp.com/2857003/api/v1/projects/16664751/todolists/"+str(todolist)+"/todos.json"
    payload = "{\n\t\"content\":\""+ title+"\""+",\n\t\"due_at\":\"2019-04-30\"\n}"
    
    headers = {
        'User-Agent': "Basecamp ClickRay integration (clickray@gmail.com)",
        'Authorization': "Basic Ym90QGNsaWNrcmF5LnBsOkJpbmxhZGVuLjE=",
        'Content-Type': "application/json",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "e98414c7-69dc-43f6-866e-e26a8e6e8d60,a81f35d6-4acb-4707-8780-738038a1fb66",
        'Host': "basecamp.com",
        'accept-encoding': "gzip, deflate",
        'content-length': "45",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)

    data_res=json.loads(response.text)
    data_url=data_res['app_url']
    nr_ticket=data_url.split('/')[-1]
    return nr_ticket

#służy do dodania komentarza jako User
def add_comment_user(ticket_number,text):
    url = "https://basecamp.com/2857003/api/v1/projects/16664751/todos/"+str(ticket_number)+"/comments.json"
    querystring = {"todo_id": ticket_number}
    payload = "{\n\t\"content\":\""+text+"\"\n}"
    
    headers = {
        'Authorization': "Basic dXNlckBjbGlja3JheS5wbDpCaW5sYWRlbi4xMQ==",
        'Content-Type': "application/json"
    }
    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers, params=querystring)

#służy do dodania komentarza jako Bot
def add_comment_bot(ticket_number,text):
    url = "https://basecamp.com/2857003/api/v1/projects/16664751/todos/"+str(ticket_number)+"/comments.json"
    querystring = {"todo_id": ticket_number}
    payload = "{\n\t\"content\":\""+text+"\"\n}"
    
    headers = {
        'Authorization': "Basic Ym90QGNsaWNrcmF5LnBsOkJpbmxhZGVuLjE=",
        'Content-Type': "application/json"
    }
    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers, params=querystring)

#służy do dodania komentarza ze zdjęciem
def add_comment_with_photo(ticket_number,token,filename):
    url = "https://basecamp.com/2857003/api/v1/projects/16664751/todos/"+str(ticket_number)+"/comments.json"
    
    querystring = {"todo_id": ticket_number}
    daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text="User sent a screenshot! "+daytime
    payload ="{\n\t\"content\":\""+text+"\",\n\t\"attachments\":[{\n\t\"token\":\""+token+"\",\"name\":\""+filename+"\"}]\n}"
    
    headers = {
        'Authorization': "Basic Ym90QGNsaWNrcmF5LnBsOkJpbmxhZGVuLjE=",
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
        'cache-control': "no-cache"
    }
    
    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)

#służy do zamknięcia ticketa
def complete_ticket(ticket_number):
    url = "https://basecamp.com/2857003/api/v1/projects/16664751/todos/"+str(ticket_number)+".json"
    
    payload = "{\n\t\"completed\":true\n}"
    headers = {
        'User-Agent': "Basecamp ClickRay integration (clickray@gmail.com)",
        'Authorization': "Basic Ym90QGNsaWNrcmF5LnBsOkJpbmxhZGVuLjE=",
        'Content-Type': "application/json",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "226c113f-2cce-422c-93c2-981d17f84df4,718038d8-ee95-4c7b-b282-678a3936053e",
        'Host': "basecamp.com",
        'accept-encoding': "gzip, deflate",
        'content-length': "21",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
    }
    
    response = requests.request("PUT", url, data=payload.encode('utf-8'), headers=headers)

#generuje token, używane przy przesyłaniu screenshotów
def return_token(binary):
    url = "https://basecamp.com/2857003/api/v1/attachments.json"
    headers = {
        'Authorization': "Basic Ym90QGNsaWNrcmF5LnBsOkJpbmxhZGVuLjE=",
        'Content-Type': "image/jpeg",
        'User-Agent': "PostmanRuntime/7.11.0",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "26b71915-1719-4eac-8207-f5bad49a26d9,20073059-29a5-405f-bb6d-ba378739b2ec",
        'Host': "basecamp.com",
        'accept-encoding': "gzip, deflate",
        'content-length': "682860",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=binary, headers=headers)
    d=json.loads(response.text)
    token=d["token"]
    return token

#dodaje końcowy komentarz i wysyła historie rozmowy na maila
def add_ending_comm_and_mail(ticket_number,text,user_email):
    url = "https://basecamp.com/2857003/api/v1/projects/16664751/todos/"+str(ticket_number)+"/comments.json"
    querystring = {"todo_id": ticket_number}
    payload = "{\n\t\"content\":\""+text+"\",\n\t\"new_subscriber_emails\":[\""+user_email+"\"]\n}"
    
    headers = {
        'Authorization': "Basic Ym90QGNsaWNrcmF5LnBsOkJpbmxhZGVuLjE=",
        'Content-Type': "application/json",
        'cache-control': "no-cache",
        'Postman-Token': "63d67b4e-5a53-460a-b8cc-6484b1405d89"
    }
    
    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers, params=querystring)

#pobiera avatar użytkownika z HubSpota
def get_avatar(email):
    try:
        response = urllib.request.urlopen('https://api.hubapi.com/contacts/v1/contact/email/'+email+'/profile?hapikey=4de5f154-adc5-4d3c-be1e-88137d68372d')
        myfile = response.read()
        myfile2= json.loads(myfile)
        m3=myfile2['properties']
        m4=m3['hs_avatar_filemanager_key']
        m5=m4['value']
        print(m5)
        result="http://cdn2.hubspot.net/"+m5
        return result
    except:
        return ""

#sprawdza czy dany email jest w bazie HubSpota (czy user jest klientem)
def is_a_hubspot_client(email):
    try:
        response = urllib.request.urlopen('https://api.hubapi.com/contacts/v1/contact/email/'+email+'/profile?hapikey=4de5f154-adc5-4d3c-be1e-88137d68372d&property=marketplace_template_price&propertyMode=value_only&formSubmissionMode=none&showListMemberships=false&property=lifecyclestage')
        myfile = response.read()
        myfile2= json.loads(myfile)
        m3=myfile2['properties']
        m4=m3['lifecyclestage']
        m4prim=m3['marketplace_template_price']
        m5=m4['value']
        m5prim=m4prim['value']
        if (m5=="customer" or m5prim>0):
            print(m5)
            print(m5prim)
            return True
    except:
        try:
            response = urllib.request.urlopen('https://api.hubapi.com/contacts/v1/contact/email/'+email+'/profile?hapikey=4de5f154-adc5-4d3c-be1e-88137d68372d&property=marketplace_template_price&propertyMode=value_only&formSubmissionMode=none&showListMemberships=false&property=lifecyclestage')
            myfile = response.read()
            myfile2= json.loads(myfile)
            m3=myfile2['properties']
            m4=m3['lifecyclestage']
            m5=m4['value']
            if (m5=="customer"):
                return True
        except:
            return False


#import danych z pliku - bazy wiedzy
def import_data():
    
    #z JSONA pobieramy 10 rzeczy
    
    source=pd.DataFrame(columns=['Tag','Question', 'Answer', 'Category','Module_Template','Src_1','Src_2','Url_1','Url_2','Url_3'])
    
    tags=[]
    questions=[]
    answers=[]
    categories=[]
    modules=[]
    src1=[]
    src2=[]
    url1=[]
    url2=[]
    url3=[]
    
    GREETING_INPUTS =[]
    GREETING_RESPONSES = ["hi", "hey", "hi there", "hello"]
    CARE_INPUTS = ("how are you", "what's up", "how R U","how are you?", "what's up?", "how R U?")
    CARE_RESPONSES = ["Good, thanks! How can I help you?", "Excellent, thanks! Feel free to asking me questions."]
    
    with open('source2new.json') as f:
        data=json.load(f)
        for p in data['rows']:
            tags.append(p['tag'])
            questions.append(p['question'])
            answers.append(p['answer'])
            categories.append(p['category'])
            modules.append(p['module_template'])
            src1.append(p['src_1'])
            src2.append(p['src_2'])
            url1.append(p['url_1'])
            url2.append(p['url_2'])
            url3.append(p['url_3'])


    source['Tag']=pd.Series(tags)
    source['Question']=pd.Series(questions)
    source['Answer']=pd.Series(answers)
    source['Category']=pd.Series(categories)
    source['Module_Template']=pd.Series(modules)
    source['Src_1']=pd.Series(src1)
    source['Src_2']=pd.Series(src2)
    source['Url_1']=pd.Series(url1)
    source['Url_2']=pd.Series(url2)
    source['Url_3']=pd.Series(url3)

    #dokonujemy obróbki danych, lemmatyzujemy, tokenizujemy itd

    questions2=deepcopy(questions)
    for q in questions2:
        q=q.lower()
    
    questions_list=''
    for q in questions2:
        questions_list=questions_list+' '+q

    q_sent_tokens=nltk.sent_tokenize(questions_list)
    q_word_tokens=nltk.word_tokenize(questions_list)

    answers2=deepcopy(answers)
    for a in answers2:
        a=a.lower()
    
    answers_list=''
    for a in answers2:
        answers_list=answers_list+' '+a

    a_sent_tokens=nltk.sent_tokenize(answers_list)
    a_word_tokens=nltk.word_tokenize(answers_list)

    tags2=deepcopy(tags)
    for t in tags2:
        t=t.lower()

    tags_list=''
    for t in tags2:
        tags_list=tags_list+' '+t
    
    t_sent_tokens=nltk.sent_tokenize(tags_list)
    t_word_tokens=nltk.word_tokenize(tags_list)


    lemmer = nltk.stem.WordNetLemmatizer()

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    
    daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #zwracamy obrobione dane
    
    return tags,questions,answers,categories,t_sent_tokens,q_sent_tokens,a_sent_tokens,tags_list,questions_list,answers_list,src1,src2,url1,url2,url3

#następuje import danych
tags,questions,answers,categories,t_sent_tokens,q_sent_tokens,a_sent_tokens,tags_list,questions_list,answers_list,src1,src2,url1,url2,url3=import_data()

#tworzymy klasyfikator pytań
global our_pipeline
our_pipeline=question_classifier()

#w tej funkcji szukamy odpowiedzi dla użytkownika zalogowanego - czyli ta funkcja zawiera dodatkowe facilities jak łączenie z basecampem i możliwosc przesyłania pytania do Biura Obsługi Klienta
def find_answer(id, problem, t_sent_tokens,q_sent_tokens, a_sent_tokens, tags, questions, answers, categories, ticket_number, src1, src2, url1,url2,url3,name):
    
    tags,questions,answers,categories,t_sent_tokens,q_sent_tokens,a_sent_tokens,tags_list,questions_list,answers_list,src1,src2,url1,url2,url3=import_data()

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    user_response=deepcopy(problem)
    user_response=user_response.lower()
    
    #Jeśli user chciał się tylko przywitac, to zostanie również przywitany
    #Jeśli chciał zapytac "Co słychać?" to dostanie odpowiedź
    #Jeśli chciał zapytać która godzina, to dostanie odpowiedź z godziną
    
    GREETING_INPUTS=["hi", "hey", "hi there", "hello","hi bot", "hello bot"]
    GREETING_RESPONSES = ["Hi!", "Hey!", "Hi there!", "Hello!"]
    CARE_INPUTS = ["how are you", "what's up", "how R U","how are you?", "what's up?", "how R U?"]
    CARE_RESPONSES = ["Good, thanks! How can I help you?", "Excellent, thanks! Feel free to asking me questions."]
    TIME_INPUTS=["what time is it?","what time is it", "what time is now?","what's the time?","what's the time","whats the time?","whats the time"]
    
    if problem in GREETING_INPUTS:
        sents=[]
        robo_response=random.choice(GREETING_RESPONSES)+" Feel free to asking me questions about your problem."
        add_comment_bot(ticket_number,robo_response)
        text2="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message\": \""+robo_response+"\",\"messageType\": 1}"
        return sents,text2

    if problem in CARE_INPUTS:
        sents=[]
        robo_response=random.choice(CARE_RESPONSES)
        add_comment_bot(ticket_number,robo_response)
        text2="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message\": \""+robo_response+"\",\"messageType\": 1}"
        return sents,text2

    if problem in TIME_INPUTS:
        daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sents=[]
        robo_response="It's "+daytime+" :)"
        add_comment_bot(ticket_number,robo_response)
        text2="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message\": \""+robo_response+"\",\"messageType\": 1}"
        return sents,text2


    robo_response=''
    robo_response2=''
    
    #Badamy podobieństwo zadanego pytania do pytań z bazy i zapisujemy podobne do tablicy
    
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

    #Badamy podobieństwo zadanego pytania do odpowiedzi z bazy i zapisujemy podobne do tablicy


    a_sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_a = TfidfVec.fit_transform(a_sent_tokens)

    vals_a = cosine_similarity(tfidf_a[-1], tfidf_a)
    propos_a=[]
    iters=0
    
    flat_a = vals_a.flatten()
    flat_a.sort()
    flat_a= np.delete(flat_a, -1)
    idx_a=vals_a.argsort()
    idx_a= np.delete(idx_a, -1)
    idx_a=list(idx_a)
    
    for i in range(len(flat_a)):
        if (flat_a[i]>0.2):
            propos_a.append(idx_a[i])
            iters=+1


    #Badamy podobieństwo zadanego pytania do tagów z bazy i zapisujemy podobne do tablicy

    t_sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_t = TfidfVec.fit_transform(t_sent_tokens)

    vals_t= cosine_similarity(tfidf_t[-1], tfidf_t)
    propos_t=[]
    iters=0
    
    flat_t= vals_t.flatten()
    flat_t.sort()
    flat_t= np.delete(flat_t, -1)
    idx_t=vals_t.argsort()
    idx_t= np.delete(idx_t, -1)
    idx_t=list(idx_t)
    
    for i in range(len(flat_t)):
        if (flat_t[i]>0.5):
            propos_t.append(idx_t[i])
            iters=+1



    sents=[]
    cats=[]
    iters2=0
    req_tfidf = flat_a[-1]

    #Jeśli nie zostało wykryte żadne podobieństwo, zwracamy informację o nierozpoznaniu pytania i mozliwosci przesłania pytania do BOK
    
    
    if (req_tfidf==0 or (len(propos_a)==0 and len(propos_q)==0 and len(propos_t)==0)):
        robo_response="I'm sorry, but I don't understand you. Please, reword the question or ask another one. Or if you want to send your query to the Customer Service, click 'send'. This is also the right place to insert a screenshot if needed - just put it here!"
        robo_response_22="And if you're looking for another way, maybe you would like to have a look at our FAQ, that may help in finding solutions."
        link="https://clickray.eu/faq/"
        add_comment_bot(ticket_number,robo_response)
        add_comment_bot(ticket_number,robo_response_22)
        add_comment_bot(ticket_number,link)
        text="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message_2\": \"And if you're looking for another way, maybe you would like to have a look at our FAQ, that may help in finding solutions.\",\"message\": \""+ robo_response+"\",\"messageType\": 2,\"messageType_2\": 1,\"images_2\": {\"url_2\": \""+link+"\"},\"messageOptions\": {\"opt_s\": \"SEND!\"}}"
        return sents,text


    else:
        
        #Pytanie rozpoznane, generujemy listę podobnych pytań do wyboru dla usera
        
        
        if len(propos_t)!=0:
            for i in range(len(propos_t)):
                idxx=idx_t[-i-1]
                t_sent_tokens[idxx]=t_sent_tokens[idxx].lstrip()
                if t_sent_tokens[idxx] in sents:
                    robo_response = robo_response
                else:
                    sents.append(t_sent_tokens[idxx])
                    cats.append(categories[idxx])
    
        if len(propos_a)!=0:
            for i in propos_a:
                for a in answers:
                    a=str(a)
                    if a_sent_tokens[i] in a:
                        #if i in a:
                        idxx=answers.index(a)
                        if t_sent_tokens[idxx] in sents:
                            robo_response = robo_response
                        else:
                            sents.append(t_sent_tokens[idxx])
                            cats.append(categories[idxx])

        if len(propos_q)!=0:
            for i in propos_q:
                for q in questions:
                    q=str(q)
                    if q_sent_tokens[i] in q:
                        
                        idxx=questions.index(q)
                        if t_sent_tokens[idxx] in sents:
                            robo_response = robo_response
                        else:
                            sents.append(t_sent_tokens[idxx])
                            cats.append(categories[idxx])

        for s in sents:
            s=s.lstrip()
        
        if len(sents)==0:
            print("ERROR! Len(sents)==0")

        l=[]
        l.append(problem)

        #Rozpoznajemy kategorie wprowadzonego pytania

        p=our_pipeline.predict(l)

        t2="With the help of artificial intelligence, I predict that your question comes from the category: "+p[0]

        add_comment_bot(ticket_number,t2)

        #Zwracamy odpowiedź Userowi - listę najbardziej podobnych pytań, całość archiwizujemy w BaseCampie
    
        text="(number/no) Are you looking for information about: "+"<br/>"
        text2="Are you looking for information about: "

        text_karol="{\"id\": \"" +str(id) + "\",\"message\": \""+t2+"\",\"helpdesk-name\": \""+name+"\",\"endingFlag\": true,\"emailInput\": true,\"message_2\": \"" + text2 + "\",\"messageInput\": true,\"messageType\": 1,\"messageType_2\": 2,\"messageOptions_2\": {"

        t=deepcopy(text)
        tk=deepcopy(text_karol)

        
        for i in range(len(sents)):
            ind=tags.index(sents[i])
            text=str(i+1)+". "+sents[i]+"? (category: "+cats[i]+")"
            t=t+text+"<br/>"
            
            text_karol="\"opt_"+ str(i+1) + "\": \"" + str(i+1)+". "+sents[i]+"? (category: "+cats[i]+")" +"\","
            
            tk=tk+text_karol

        tk=tk+"\"opt_non\": \"None of the above?\"}}"
        t=t+", None of the above?"

        add_comment_bot(ticket_number,t)
        return sents,tk

#W tej funkcji odbywa się szukanie odpowiedzi dla użytkownika niezalogowanego, wersja uboższa

def find_answer2(id, problem, t_sent_tokens, q_sent_tokens, a_sent_tokens,tags, questions, answers, categories, src1, src2, url1,url2,url3,name):
    
    tags,questions,answers,categories,t_sent_tokens,q_sent_tokens,a_sent_tokens,tags_list,questions_list,answers_list,src1,src2,url1,url2,url3=import_data()
    
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    user_response=deepcopy(problem)
    user_response=user_response.lower()
    
    #Jeśli user chciał się tylko przywitac, to zostanie również przywitany
    #Jeśli chciał zapytac "Co słychać?" to dostanie odpowiedź
    #Jeśli chciał zapytać która godzina, to dostanie odpowiedź z godziną
    
    GREETING_INPUTS=["hi", "hey", "hi there", "hello","hi bot", "hello bot"]
    GREETING_RESPONSES = ["Hi!", "Hey!", "Hi there!", "Hello!"]
    CARE_INPUTS = ["how are you", "what's up", "how R U","how are you?", "what's up?", "how R U?"]
    CARE_RESPONSES = ["Good, thanks! How can I help you?", "Excellent, thanks! Feel free to asking me questions. &#128512; "]
    TIME_INPUTS=["what time is it?","what time is it", "what time is now?","what's the time?","what's the time","whats the time?","whats the time"]
    
    if problem in GREETING_INPUTS:
        sents=[]
        text=random.choice(GREETING_RESPONSES)+" Feel free to asking me questions about your problem."
        robo_response=deepcopy(text)
        text2="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message\": \""+ text+"\",\"messageType\": 1}"
        return sents,text2
    
    if problem in CARE_INPUTS:
        sents=[]
        text=random.choice(CARE_RESPONSES)
        robo_response=deepcopy(text)
        text2="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message\": \""+text+"\",\"messageType\": 1}"
        return sents,text2
    
    if problem in TIME_INPUTS:
        daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sents=[]
        text="It's "+daytime
        robo_response=deepcopy(text)
        text2="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message\": \""+text+"\",\"messageType\": 1}"
        return sents,text2
    
    
    robo_response=''
    robo_response2=''
    
    #Badamy podobieństwo zadanego pytania do pytań z bazy i zapisujemy podobne do tablicy
    
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
        if (flat_q[i]>0.5):
            propos_q.append(idx_q[i])
            
            iters=+1

    #Badamy podobieństwo zadanego pytania do odpowiedzi z bazy i zapisujemy podobne do tablicy


    a_sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_a = TfidfVec.fit_transform(a_sent_tokens)

    vals_a = cosine_similarity(tfidf_a[-1], tfidf_a)
    propos_a=[]
    iters=0
        
    flat_a = vals_a.flatten()
    flat_a.sort()
    flat_a= np.delete(flat_a, -1)
    idx_a=vals_a.argsort()
    idx_a= np.delete(idx_a, -1)
    idx_a=list(idx_a)
        
    for i in range(len(flat_a)):
        if (flat_a[i]>0.2):
            propos_a.append(idx_a[i])
            iters=+1


    #Badamy podobieństwo zadanego pytania do tagów z bazy i zapisujemy podobne do tablicy

    t_sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_t = TfidfVec.fit_transform(t_sent_tokens)

    vals_t= cosine_similarity(tfidf_t[-1], tfidf_t)
    propos_t=[]
    iters=0
    
    flat_t= vals_t.flatten()
    flat_t.sort()
    flat_t= np.delete(flat_t, -1)
    idx_t=vals_t.argsort()
    idx_t= np.delete(idx_t, -1)
    idx_t=list(idx_t)
    
    for i in range(len(flat_t)):
        if (flat_t[i]>0.2):
            propos_t.append(idx_t[i])
            iters=+1

    sents=[]
    cats=[]
    iters2=0
    req_tfidf = flat_a[-1]

    #Jeśli nie zostało wykryte żadne podobieństwo, zwracamy informację o nierozpoznaniu pytania i mozliwosci przesłania pytania do BOK
    
        
    if (req_tfidf==0 or (len(propos_a)==0 and len(propos_q)==0 and len(propos_t)==0)):
        robo_response="I'm sorry, but I don't understand you. Please, reword the question or ask another one."
        robo_response_22="And if you're looking for another way, maybe you would like to have a look at our FAQ, that may help in finding solutions."
        link="https://clickray.eu/faq/"
        text="{\"id\": \""+str(id)+ "\",\"helpdesk-name\": \""+name+"\",\"emailInput\": true,\"message_2\": \"And if you're looking for another way, maybe you would like to have a look at our FAQ, that may help in finding solutions.\",\"message\": \""+ robo_response+"\",\"messageType\": 1,\"messageType_2\": 1,\"images_2\": {\"url_2\": \""+link+"\"}}"
        
        return sents,text

    #Pytanie rozpoznane, generujemy listę podobnych pytań do wyboru dla usera

    else:
        if len(propos_t)!=0:
            for i in range(len(propos_t)):
                idxx=idx_t[-i-1]
                t_sent_tokens[idxx]=t_sent_tokens[idxx].lstrip()
                if t_sent_tokens[idxx] in sents:
                    robo_response = robo_response
                else:
                    sents.append(t_sent_tokens[idxx])
                    cats.append(categories[idxx])

        if len(propos_a)!=0:
            for i in propos_a:
                for a in answers:
                    a=str(a)
                    if a_sent_tokens[i] in a:
                        idxx=answers.index(a)
                        if t_sent_tokens[idxx] in sents:
                            robo_response = robo_response
                        else:
                            sents.append(t_sent_tokens[idxx])
                            cats.append(categories[idxx])

        if len(propos_q)!=0:
            for i in propos_q:
                for q in questions:
                    q=str(q)
                    if q_sent_tokens[i] in q:
                        idxx=questions.index(q)
                        if t_sent_tokens[idxx] in sents:
                            robo_response = robo_response
                        else:
                            sents.append(t_sent_tokens[idxx])
                            cats.append(categories[idxx])
                         
    for s in sents:
        s=s.lstrip()
            
    if len(sents)==0:
        print("ERROR! Len(sents)==0")

    #Rozpoznajemy kategorie wprowadzonego pytania
    
    l=[]
    l.append(problem)
    p=our_pipeline.predict(l)
        
    t2="With the help of artificial intelligence, I predict that your question comes from the category: "+p[0]

    #Zwracamy odpowiedź Userowi - listę najbardziej podobnych pytań

    text="(number/no) Are you looking for information about: "
    text2="Are you looking for information about: "+"<br/>"
            
    text_karol="{\"id\": \"" +str(id) + "\",\"message\": \""+t2+"\",\"helpdesk-name\": \""+name+"\",\"endingFlag\": true,\"emailInput\": true,\"message_2\": \"" + text2 + "\",\"messageInput\": true,\"messageType\": 1,\"messageType_2\": 2,\"messageOptions_2\": {"
            
    t=deepcopy(text)
    tk=deepcopy(text_karol)
            
            
    for i in range(len(sents)):
        ind=tags.index(sents[i])
        text=str(i+1)+". "+sents[i]+"? (category: "+cats[i]+")"
        t=t+text+"<br/>"
                
        text_karol="\"opt_"+ str(i+1) + "\": \"" + str(i+1)+". "+sents[i]+"? (category: "+cats[i]+")" +"\","
                
        tk=tk+text_karol

    tk=tk+"\"opt_non\": \"None of the above?\"}}"
    t=t+", None of the above?"
    return sents,tk


tn=0
opt=0
sents=[]

#w ponizszej części znajduje się backend aplikacji, interakcja serwera z klientem

class Handler(BaseHTTPRequestHandler):
    
    
    def do_OPTIONS(self):
        print("OPTION")
        
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
        form = cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={'REQUEST_METHOD': 'POST'})
        print(form)
        print(type(form))
        
        return
    
    
    
    def do_GET(self):
        print("GET")
        self.send_response(200)
        self.end_headers()
        return


    def do_POST(self):
        print("POST")
        print(self.client_address)
        global tn
        global opt
        global sents
  
        #wczytujemy strumień danych przesłany przez klienta
        
        form=cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={'REQUEST_METHOD': 'POST'})
        
        #wczytujemy zmienne z wiadomości od klienta
        
        name = form.getfirst("helpdesk-name", "")
        send_b=form.getfirst("opt_s", "")
        mess=form.getfirst("message", "")
        satY=form.getfirst("opt_yes_sat","")
        satN=form.getfirst("opt_no_sat","")
        seeY=form.getfirst("opt_yes_see","")
        seeN=form.getfirst("opt_no_see","")
        bye=form.getfirst("opt_bye","")
        src_1=form.getfirst("src_1","")
        src_2=form.getfirst("src_2","")
        url_1=form.getfirst("url_1","")
        url_2=form.getfirst("url_2","")
        url_3=form.getfirst("url_3","")
        data_url=form.getfirst("dataUrl","")
        no=form.getfirst("opt_no","")
        sel=form.getfirst("selected","")
        em_input=form.getfirst("emailInput","")
        non=form.getfirst("opt_non","")
        selected_op=form.getfirst("selected-option","")
        problem = form.getfirst("helpdesk-problem", "")
        
        form_sel=form.getfirst("select", "")
        form_firstname=form.getfirst("First Name*", "")
        form_lastname=form.getfirst("Last Name*", "")
        form_email=form.getfirst("Email*", "")
        form_company=form.getfirst("Company Name*", "")
        form_mess=form.getfirst("hbmessage", "")
        
        form_paypal=form.getfirst("Email for Paypal request", "")
        form_subject=form.getfirst("Subject", "")
        form_template=form.getfirst("Template Name", "")
        form_url=form.getfirst("Page URL", "")
        form_website=form.getfirst("Website", "")
        form_streetname=form.getfirst("Street name*", "")
        form_streetnumber=form.getfirst("Street number*", "")
        form_streetlocalnumber=form.getfirst("Local number*", "")
        form_zipcode=form.getfirst("ZIP code*", "")
        form_city=form.getfirst("City*", "")
        form_state=form.getfirst("State*", "")
        form_country=form.getfirst("Country*", "")
        form_taxid=form.getfirst("Tax ID number*", "")
        form_paypaltrans=form.getfirst("Paypal transaction ID*", "")
        form_item=form.getfirst("Item purchased*", "")
        form_amount=form.getfirst("Amount spent*", "")
        
        #obrabiamy/oczyszczamy dane od klienta
        
        name=name.rstrip()
        mess=mess.rstrip()
        form_mess=form_mess.rstrip()
    
        name.replace(u'\u2019', '-')
        mess.replace(u'\u2019', '-')
        form_mess.replace(u'\u2019', '-')
        
        name.encode('latin-1', 'replace')
        mess.encode('latin-1', 'replace')
        form_mess.encode('latin-1', 'replace')
        
        name.encode('UTF8', 'replace')
        mess.encode('UTF8', 'replace')
        form_mess.encode('UTF8', 'replace')
        
        name.encode('UTF8', 'replace')
        mess.encode('UTF8', 'replace')
        form_mess.encode('UTF8', 'replace')
        
        #przechwytujemy SPAM
        
        if len(name)!=0:
            print("FORM: ",form)
            formId = form.getfirst("formId", "")
            if len(formId)==0:
                email = form.getfirst("helpdesk-email", "")
                if len(email)!=0:
                    add_comment_bot("393099535",email)
                else:
                    add_comment_bot("393099535","Proba wejscia bez podawania maila")
    
    #poniżej znajduje się switch reakcji bota, na podstawie danych przesyłanych przez klienta
        
        if len(sel)!=0:
         #tu wskakujemy, gdy User jest na "stronie głównej" bota
            formId = form.getfirst("formId", "")
            if len(formId)==0:
                
                #tu wchodzimy, jeśli User zaczyna inteakcje z Botem, tworzymy Usera "person" i dodajemy go do tablicy
                #ponizej są cechy Usera
            
                person=[]
                global id
                person.append(id) #0 id
                person.append("") #1 name
                person.append("") #2 mail
                person.append("") #3 question
                person.append(0) # 4 ticket number
                person.append([]) #5 to lista propozycji
                person.append("true") #6 czy odpowiedz udzielona
                person.append(sel) #7 helpdesk czy purchase
                person.append("") #8 selected option
            
                threads.append(person)
                
            
                formId=deepcopy(id)
                id=id+1
            
                print("Threads: ",threads)
            
                message="{\"id\": \""+str(formId)+"\",\"endingFlag\": false}"
    
            else:
                #tu wskakujemy, gdy User już jest utworzony a jedynie przechodzi między głównymi trzema zakładkami na stronie głównej
                #również tu aktualizujemy mu flagę zalogowany/niezalogowany
                
                for i in range(len(threads)):
                    if int(threads[i][0])==int(formId):
                        tn=deepcopy(threads[i][4])
                        sel_option=deepcopy(threads[i][7])
                        email=deepcopy(threads[i][2])
                        if sel_option=="purchase":
                            if (sel=="helpdesk" and len(email)!=0):
                                threads[i][7]="helpdesk"
                            if (sel=="helpdesk" and len(email)==0):
                                threads[i][7]="purchase"
                    
                        
                        if sel_option=="agencies":
                            if sel=="helpdesk":
                                threads[i][7]="helpdesk"
                        
                        if sel_option=="agencies":
                            if sel=="purchase":
                                threads[i][7]="purchase"
    
                        if sel=="agencies":
                            threads[i][8]="10"
            
            
                print(threads)
                message="{\"id\": \""+str(formId)+"\",\"endingFlag\": false}"

        
        if len(name)!=0:
            
            #tu User się loguje, uzupełniamy dane w tablicy informacji o nim
            
            formId = form.getfirst("formId", "")
            formId=int(formId)
            
            for i in range(len(threads)):
                if threads[i][0]==formId:
                    tn=deepcopy(threads[i][4])
                    threads[i][6]="false"
            
            formId = form.getfirst("formId", "")
            formId=int(formId)
            
            problem = form.getfirst("helpdesk-problem", "")
            problem=problem.rstrip()
            problem.replace(u'\u2019', '-')
            problem.encode('latin-1', 'replace')
            problem.encode('utf-8')
            
            resp=deepcopy(problem)
            resp=resp.replace('"','!d_quote!')
            resp=resp.replace('\'','!s_quote!')
            resp.encode('utf-8')
            
            resp2=resp[0:25]
            resp2.replace(u'\u2019', '-')
            resp2.encode('latin-1', 'replace')
    
            
            email = form.getfirst("helpdesk-email", "")
            if len(email)!=0:
                email.replace(u'\u2019', '-')
                email.encode('latin-1', 'replace')
                email.encode('utf-8')
                
                if is_a_hubspot_client(email) == True:
                    ticket_number=make_ticket(name,email,resp2,"57151473")
                    for i in range(len(threads)):
                        if threads[i][0]==formId:
                            threads[i][1]=name
                            threads[i][2]=email
                            threads[i][3]=problem
                            threads[i][4]=ticket_number
        
                    #Jeśli jest zalogowany, dodajemy w BaseCampie informacje o zgodach
        
                    rodo="The User has accepted the Privacy Policy of Clickray sp. z o. o. and has consented to the storage and processing of his/her personal data."
                    cust="The User has been identified as a Hubspot customer."
            
                    add_comment_bot(ticket_number,rodo)
                    add_comment_bot(ticket_number,cust)
                    add_comment_user(ticket_number,problem)
                
                    print(threads)
                    avatar_url=get_avatar(email)
                    name2=name.title()
                    message="{\"id\": \""+str(formId)+"\",\"emailInput\": true,\"helpdeskName\": \""+name2+"\",\"avatarPhoto\":\""+avatar_url+"\",\"endingFlag\": false}"
                    
                if is_a_hubspot_client(email) == False:
                    message="{\"id\": \""+str(formId)+"\",\"emailInput\": false,\"messageType\": 1,\"endingFlag\": false}"
                            
            else:
                for i in range(len(threads)):
                    if threads[i][0]==formId:
                        threads[i][1]=name
                        threads[i][3]=problem
                        message="{\"id\": \""+str(formId)+"\"}"
                
        if len(selected_op)!=0:
            #Tu reakcja gdy User przechodzi między podstronami, jeśli jest zalogowany to zostawiamy informację o tym w BaseCampie
            formId = form.getfirst("formId", "")
            formId=int(formId)
            
            for i in range(len(threads)):
                if threads[i][0]==formId:
                    tn=deepcopy(threads[i][4])
                    it=deepcopy(i)
                    threads[i][8]=deepcopy(selected_op)
                    permision=deepcopy(threads[i][7])
            
            if permision=="helpdesk":
                text="User chose the path: Purchase -> Option "+str(selected_op)
                add_comment_user(tn,text)
    
            message="{\"id\": \""+str(formId)+"\"}"
            
        if (len(form_firstname)!=0 or len(form_sel)!=0 or len(form_lastname)!=0 or len(form_email)!=0 or len(form_company)!=0):
            #Tu jeśli User wypełnił właśnie któryś formularz, przechwytujemy dane z HubSpota i zapisujemy w Basecampie tworząc na to osobny ticket
            formId = form.getfirst("formId", "")
            formId=int(formId)
            
            for i in range(len(threads)):
                if threads[i][0]==formId:
                    tn=deepcopy(threads[i][4])
                    it=deepcopy(i)
                    permision=deepcopy(threads[i][7])
                    selected_op=deepcopy(threads[i][8])
        
            
            if (len(form_firstname)!=0 and len(form_lastname)!=0 and len(form_email)!=0):
                text="Selected Option of Form: "+str(selected_op)+"<br/>Select: "+form_sel+"<br/>First Name: "+form_firstname+"<br/>Last Name: "+form_lastname+"<br/>Email: "+form_email+"<br/>Email for PayPal request: "+form_paypal+"<br/>Company Name: "+form_company+"<br/>Template name: "+form_template+"<br/>Page URL: "+form_url+"<br/>Subject: "+form_subject+"<br/>Message: "+form_mess+"<br/>Website: "+form_website+"<br/>Street name: "+form_streetname+"<br/>Street number: "+form_streetnumber+"<br/>Local number: "+form_streetlocalnumber+"<br/>ZIP code: "+form_zipcode+"<br/>City: "+form_city+"<br/>State: "+form_state+"<br/>Country: "+form_country+"<br/>Tax ID number: "+form_taxid+"<br/>Paypal transaction ID: "+form_paypaltrans+"<br/>Item purchased: "+form_item+"<br/>Amount spent: "+form_amount
                
                if permision=="helpdesk":
                    if int(selected_op)==1:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57475909")
                    if int(selected_op)==2:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57231508")
                    if int(selected_op)==3:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57231517")
                    if int(selected_op)==4:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57231526")
                    if int(selected_op)==5:
                        print("JESTEM")
                        form_sel="INVOICE"
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"58048741")
                    if int(selected_op)==10:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57430199")
                    add_comment_bot(ticket_number,text)
                    add_comment_user(tn,text)
                    message="{\"id\": \""+str(formId)+"\"}"
            
                if permision=="purchase":
                    if int(selected_op)==1:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57475909")
                    if int(selected_op)==2:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57231508")
                    if int(selected_op)==3:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57231517")
                    if int(selected_op)==4:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57231526")
                    if int(selected_op)==5:
                        print("JESTEM")
                        form_sel="INVOICE"
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"58048741")
                    if int(selected_op)==10:
                        ticket_number=make_ticket(form_lastname,form_email,form_sel,"57430199")
                    add_comment_bot(ticket_number,text)
                    message="{\"id\": \""+str(formId)+"\"}"
                        
                if permision=="agencies":
                    ticket_number=make_ticket(form_lastname,form_email,form_sel,"57430199")
                    add_comment_bot(ticket_number,text)
                    message="{\"id\": \""+str(formId)+"\"}"
                        
                

            else:
                message="{\"id\": \""+str(formId)+"\"}"

                                
        
        if len(send_b)!=0:
            #Tu gdy User chce przesłać swoje pytanie do Biura Obsługi Klienta
            
            formId = form.getfirst("formId", "")
            formId=int(formId)
            
            for i in range(len(threads)):
                if threads[i][0]==formId:
                    tn=deepcopy(threads[i][4])
                    threads[i][6]="false"
        
            daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            text=send_b+" "+daytime
            add_comment_user(tn,text)
            robo_response2="We forward your question to the Customer Service Department, we will contact with you within 48 hours. Ask another question if you want or click 'Bye'. You can also send a screenshot - just put it here!"
            add_comment_bot(tn,robo_response2)
            message="{\"id\": \""+str(formId)+"\",\"message\": \""+robo_response2+"\",\"messageType\": 3, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": false}"
        
        if len(mess)!=0:
            #Tu następuje generowanie odpowiedzi na pytanie Usera po uzupełnieniu formularza kontaktowego
            mess = form.getfirst("message", "")

            formId = form.getfirst("formId", "")
            formId=int(formId)
            for i in range(len(threads)):
                if int(threads[i][0])==formId:
                    tn=deepcopy(threads[i][4])
                    threads[i][3]=deepcopy(mess)
                    name=deepcopy(threads[i][1])
                    permision=deepcopy(threads[i][7])
        
            mess2=deepcopy(mess)
            mess2=mess2.replace('"','!d_quote!')
            mess2=mess2.replace('\'','!s_quote!')
            
            if permision=="helpdesk":

                text=deepcopy(mess2)
                add_comment_user(tn,text)
                sents,message=find_answer(formId, mess,t_sent_tokens, q_sent_tokens, a_sent_tokens,tags, questions, answers, categories, tn,src1,src2,url1,url2,url3,name)
            
            
            if permision=="purchase":
                sents,message=find_answer2(formId, mess,t_sent_tokens, q_sent_tokens, a_sent_tokens,tags, questions, answers, categories,src1,src2,url1,url2,url3,name)
            
        
            for i in range(len(threads)):
                if threads[i][0]==formId:
                    threads[i][5]=deepcopy(sents)

        if len(problem)!=0:
            #Tu gdy nie jest zalogowany i wpisuje problem poza formularzem
            formId = form.getfirst("formId", "")
            formId=int(formId)
            for i in range(len(threads)):
                if int(threads[i][0])==formId:
                    purchase_problem=deepcopy(threads[i][3])
                    permision=deepcopy(threads[i][7])
                    name=deepcopy(threads[i][1])

            if permision=="purchase":
                sents,message=find_answer2(formId, purchase_problem, t_sent_tokens,q_sent_tokens, a_sent_tokens,tags, questions, answers, categories,src1,src2,url1,url2,url3,name)
                    
            for i in range(len(threads)):
                if threads[i][0]==formId:
                    threads[i][5]=deepcopy(sents)


        if len(no)!=0 or len(non)!=0:
            #Tu jeśli pytanie nie zostało rozpoznane
            print("FORM: ",form)
            formId = form.getfirst("formId", "")
            formId=int(formId)
            for i in range(len(threads)):
                if int(threads[i][0])==formId:
                    tn=deepcopy(threads[i][4])
                    permision=deepcopy(threads[i][7])


            if permision=="helpdesk":
        
                rr="None of the above!"
                add_comment_user(tn,rr)
        
                robo_response="I am sorry! If none of my answers satisfy you, try asking your question differently or ask another question. Or, if you want me to send your question to the Customer Service Department, click 'send'. You can also send a screenshot - just put it here!"
                add_comment_bot(tn,robo_response)
                message="{\"id\": \""+str(formId)+"\",\"message\": \""+ robo_response+"\",\"messageType\": 2,\"messageOptions\": {\"opt_s\": \"SEND!\"},\"endingFlag\": false}"

            if permision=="purchase":
                robo_response2="I am sorry! If none of my answers satisfy you, try asking your question differently or ask another question."
                message="{\"id\": \""+str(formId)+"\",\"message\": \""+ robo_response2+"\",\"messageType\": 1,\"endingFlag\": false}"
                
    
            
        if len(data_url)!=0:
            #Tu jeśli użytkownik przesłał screenshot, dodajemy go do basecampa
            formId = form.getfirst("formId", "")
            formId=int(formId)
            for i in range(len(threads)):
                if int(threads[i][0])==formId:
                    tn=deepcopy(threads[i][4])
                    permision=deepcopy(threads[i][7])

            if permision=="helpdesk":

                decoded = base64.b64decode(data_url)
            
                with open("myfile.bin", 'wb') as f:
                    f.write(decoded)

                with open("myfile.bin", mode='rb') as file: # b is important -> binary
                    decoded2 = file.read()
            
                token=return_token(decoded2)
    
                text="User sent photo!"
            
                filename = form.getfirst("name", "")
            
                add_comment_with_photo(tn,token,filename)

                robo_response="We forward your screenshot to the Customer Service Department, in the near future our employee will contact you. Ask another question if you want or click 'Bye'. You can also send a screenshot - just put it here!"
                robo_response2="We forward your screenshot to the Customer Service Department, in the near future our employee will contact you. Ask another question if you want or click 'Bye'. You can also send a screenshot - just put it here!"
                add_comment_bot(tn,robo_response)
                message="{\"id\": \""+str(formId)+"\",\"message\": \""+robo_response2+"\",\"messageType\": 2, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": false}"

            if permision=="purchase":
                robo_response2="We can not forward your screenshot if you are not our customer. Ask another question if you want or click 'Bye'."
                message="{\"id\": \""+str(formId)+"\",\"message\": \""+robo_response2+"\",\"messageType\": 2, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": false}"
    
    
        if len(selected_op)!=0:
            #Tu obsługujemy zwykłe przechodzenie między podstronami bez interakcji z botem
            formId = form.getfirst("formId", "")
            formId=int(formId)
            message="{\"id\": \""+str(formId)+"\",\"endingFlag\": false}"
        
        
        if len(form)==1:
            #Tu gdy ktoś wyśle formularz bez problemu
            daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if len(sel)==0:
                formId = form.getfirst("formId", "")
                formId=int(formId)
                for i in range(len(threads)):
                    if int(threads[i][0])==formId:
                        tn=deepcopy(threads[i][4])
                        permision=deepcopy(threads[i][7])
                        if len(mess)==0:
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+"Ask another question if you want or click 'Bye'."+"\",\"messageType\": 3, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": false}"
                            
                            if permision=="helpdesk":
                                robo_response="Ask another question if you want or click 'Bye'. "+daytime
                                add_comment_bot(tn,robo_response)
        
    
        if len(form)==3 or len(form)==2:
            #Tu zwraca odpowiedź na wybrane z listy pytanie
            daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            mess = form.getfirst("message", "")
            if len(mess)==0:
            
                opt=''
                
                formId = form.getfirst("formId", "")
                formId=int(formId)
                for i in range(len(threads)):
                    if int(threads[i][0])==formId:
                        tn=deepcopy(threads[i][4])
                        sents=deepcopy(threads[i][5])
                        permision=deepcopy(threads[i][7])
                        for a in range(50):
                            t="opt_"+str(a+1)
                            opt=form.getfirst(t, "")
                            if len(opt)!=0:
                                break
                
                    if len(str(opt))!=0:
                        if permision=="helpdesk":
                            kk="The user chose the option No. "+str(opt)
                            add_comment_user(tn,kk)
                    
                        opt=str(opt)
                        if opt[1].isdigit()==True:
                            opt=opt[0]+opt[1]
                        else:
                            opt=opt[0]
                        opt=int(opt)
                        ap=str(sents[int(opt)-1])
                        ap=ap.lstrip()
                        myI=tags.index(ap)
                        
                        if permision=="helpdesk":
                            anss=deepcopy(answers[myI])
                            add_comment_bot(tn,anss)
                            text="Does this answer satisfy you? (yes/no)  "+daytime
                            add_comment_bot(tn,text)

                        message="{\"id\": \""+str(formId)+"\",\"message\": \""+ answers[myI]+"\","

                        if src1[myI]!='' or src2[myI]!='' or url1[myI]!='' or url2[myI]!='':
                            message=message+"\"images\": {"
                            if src1[myI]:
                                message=message+"\"src_1\": \""+src1[myI]+"\","
                                if permision=="helpdesk":
                                    add_comment_bot(tn,src1[myI])
                            
                            if src2[myI]:
                                message=message+"\"src_2\": \""+src2[myI]+"\","
                                if permision=="helpdesk":
                                    add_comment_bot(tn,src2[myI])

                            if url1[myI]:
                                message=message+"\"url_1\": \""+url1[myI]+"\","
                                if permision=="helpdesk":
                                    add_comment_bot(tn,url1[myI])

                            if url2[myI]:
                                message=message+"\"url_2\": \""+url2[myI]+"\","
                                if permision=="helpdesk":
                                    add_comment_bot(tn,url2[myI])
                            
                            if url3[myI]:
                                message=message+"\"url_3\": \""+url3[myI]+"\","
                                if permision=="helpdesk":
                                    add_comment_bot(tn,url3[myI])

                            if message[-1]==",":
                                message=message[:-1]

                            message=message+"},"
                        message=message+"\"message_2\": \"Does this answer satisfy you?\",\"messageInput\": true,\"messageType\": 1,\"messageType_2\": 3, \"messageChoose_2\": {\"opt_yes_sat\": \"Yes\",\"opt_no_sat\": \"No\"},\"endingFlag\": true}"



        if len(satY)!=0 or len(satN)!=0:
            #Tu obsługuje odpowiedz o satysfakcji użytkownika
            daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            formId = form.getfirst("formId", "")
            formId=int(formId)
            for i in range(len(threads)):
                if int(threads[i][0])==formId:
                    print(threads)
                    ticket_number=deepcopy(threads[i][4])
                    permision=deepcopy(threads[i][7])
                    if len(satY)!=0:
                        if permision=="helpdesk":
                            t="Yes! "+daytime
                            add_comment_user(ticket_number,t)
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+"Great! Ask another question if you want or click 'Bye'. You can also send a screenshot - just put it here!"+"\",\"messageType\": 3, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": true}"
                            robo_response="Great! Ask another question if you want or click 'Bye'. You can also send a screenshot - just put it here!"
                            add_comment_bot(ticket_number,robo_response)
                        if permision=="purchase":
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+"Great! Ask another question if you want or click 'Bye'."+"\",\"messageType\": 3, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": true}"
                
                    if len(satN)!=0:
                        if permision=="helpdesk":
                            t="No! "+daytime
                            add_comment_user(ticket_number,t)
                            threads[i][6]="false"
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+"Would you like to see other tips from the list?"+"\",\"messageInput\": true,\"messageType\": 3, \"messageChoose\": {\"opt_yes_see\": \"Yes\",\"opt_no_see\": \"No\"},\"endingFlag\": true}"
                            text="Would you like to see other tips from the list? (yes/no)"
                            add_comment_bot(ticket_number,text)
                        if permision=="purchase":
                            threads[i][6]="false"
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+"Would you like to see other tips from the list?"+"\",\"messageInput\": true,\"messageType\": 3, \"messageChoose\": {\"opt_yes_see\": \"Yes\",\"opt_no_see\": \"No\"},\"endingFlag\": true}"

                            
        if len(seeY)!=0 or len(seeN)!=0:
            #Tu obsługuje odpowiedź o tym czy chce wyświetlić liste pytań jeszcze raz
            print("FORM: ",form)
            daytime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("JESTEM W SEE")
            
            formId = form.getfirst("formId", "")
            formId=int(formId)
            for i in range(len(threads)):
                if int(threads[i][0])==formId:
                    print(threads)
                    ticket_number=deepcopy(threads[i][4])
                    mess=deepcopy(threads[i][3])
                    name=deepcopy(threads[i][1])
                    permision=deepcopy(threads[i][7])
                    if len(seeY)!=0:
                        if permision=="helpdesk":
                            t="Yes! "+daytime
                            add_comment_user(ticket_number,t)
                            sents,message=find_answer(formId, mess, t_sent_tokens,q_sent_tokens, a_sent_tokens, tags,questions, answers, categories, tn,src1,src2,url1,url2,url3,name)
                                
                            for i in range(len(threads)):
                                if threads[i][0]==formId:
                                    threads[i][5]=deepcopy(sents)
                                            
                            print("Threads: ", threads)
                                        
                        if permision=="purchase":
                            sents,message=find_answer2(formId, mess,t_sent_tokens, q_sent_tokens, a_sent_tokens, tags, questions, answers, categories,src1,src2,url1,url2,url3,name)
                            for i in range(len(threads)):
                                if threads[i][0]==formId:
                                    threads[i][5]=deepcopy(sents)
            
                            print("Threads: ", threads)
                            
                            
                    if len(seeN)!=0:
                        
                        if permision=="helpdesk":
                        
                            t="No! "+daytime
                            add_comment_user(ticket_number,t)
                            threads[i][6]="false"
                            robo_response2="I forward your question to the customer service department, our employee will contact you within 48 hours. Ask another question if you want or click 'Bye'. &#128512; You can also send a screenshot - just put it here!"
                            robo_response3="I forward your question to the customer service department, our employee will contact you within 48 hours. Ask another question if you want or click 'Bye'. You can also send a screenshot - just put it here!"
                            add_comment_bot(ticket_number,robo_response2)
                            threads[i][6]="false"
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+robo_response3+"\",\"messageType\": 3, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": false}"
                                
                        if permision=="purchase":
                            threads[i][6]="false"
                            robo_response3="Unfortunately, I do not know the answer to your question. Ask another question if you want or click 'Bye'."
                            message="{\"id\": \""+str(formId)+"\",\"message\": \""+robo_response3+"\",\"messageType\": 3, \"messageChoose\": {\"opt_bye\": \"BYE!\"},\"endingFlag\": false}"
                                
        if len(bye)!=0 or len(mess)!=0:
            #Tu obługuje zakończenie rozmowy, jej archiwizacje i przesłanie kopii użytkownikowi
            if len(bye)!=0 or mess.lower()=="bye" or mess.lower()=="bye!":
                formId = form.getfirst("formId", "")
                formId=int(formId)
                for i in range(len(threads)):
                    if int(threads[i][0])==formId:
                        ticket_number=deepcopy(threads[i][4])
                        permision=deepcopy(threads[i][7])
                        
                        if permision=="helpdesk":
                        
                            ur="Bye!"
                            add_comment_user(ticket_number,ur)
                            rob="Bye! See U later!"
                            email=deepcopy(threads[i][2])
                            add_ending_comm_and_mail(ticket_number,rob,email)
                            if threads[i][6]=="true":
                                complete_ticket(ticket_number)

                            message="{\"id\": \""+str(formId)+"\",\"messageInput\": true,\"message\": \""+ rob+"\",\"messageType\": 1,\"endingFlag\": true,\"theEnd\": true}"

                        if permision=="purchase":
                            rob="Bye! See U later!"
                            message="{\"id\": \""+str(formId)+"\",\"messageInput\": true,\"message\": \""+ rob+"\",\"messageType\": 1,\"endingFlag\": true,\"theEnd\": true}"

        if len(name)!=0:
            #Anty- SPAM
            print("FORM: ",form)
            formId = form.getfirst("formId", "")
            if len(formId)==0:
                email = form.getfirst("helpdesk-email", "")
                if len(email)!=0:
                    add_comment_bot("393099535",email)
                else:
                    add_comment_bot("393099535","Proba wejscia bez podawania maila")
                        
        self.path = self.path
        print(self.path)
        
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

#Tu tworzymy serwer

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

def run():
    server = ThreadedHTTPServer(('31.172.186.53', 80), Handler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()


run()
