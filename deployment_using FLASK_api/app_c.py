from flask import Flask, jsonify, request
import numpy as np
import joblib
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow import keras
import gunicorn



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

########################################################################

# loadiding model and tokenzzer
lstm_cnn_model=tf.keras.models.load_model("CNN_b_model.h5")
token_glove=joblib.load('Token_glove.pkl')

def final_prediction(csv):
  """
  This function takes text,title,author as a string  and returns the prediction whether the news is fake or not
  """
  # creating_dictonary

  # t = dic.get('text')
  # print(type(t))


  # csv=pd.DataFrame(dic,columns=['text','title','author'])


  print('data loaded') 
  # repacing nan values with " "
  csv.fillna(' ')
  ##################################
  csv_text=csv['text'].astype('str')
  csv_title=csv['title'].astype('str')
  csv_author=csv['author'].astype('str')

  ########################    Preprocessing      ###############################

  # https://gist.github.com/sebleier/554280
  # refrence applied ai
  # we are removing the words from the stop words list: 'no', 'nor', 'not'
  stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
              "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
              'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
              'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
              'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
              'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
              'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
              's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
              've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
              'won', "won't", 'wouldn', "wouldn't"]

  def preprocessig_text(text):
    string=str(text)

    # removing url from text
    #http://urlregex.com/
    string=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ', string)
    #remove HTML from the Text column and save in the Text if present
    soup = BeautifulSoup(string, 'lxml')
    string = soup.get_text()    
    ################################################
    # decontracting text refrence applied ai
    string = re.sub(r"won't", "will not", string)
    string = re.sub(r"can\'t", "can not", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'t", " not", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'m", " am", string)
    ###################################################

    string=re.sub('(@\w{1,})','',string) # removing usernames(@johndeapth)
    string=re.sub('pic.twitter.com/.+','',string) # removing image captions
    string=re.sub('\d',' ',string) # removing digits
    string=re.sub('\n',' ', string) # repacing new line character with one space
    string=re.sub('\t',' ', string) # replacing tab with space
    ##################################################################################################
    # removing some of the spacial charcters.
    string=re.sub('\.',' ',string) # removing period
    string=re.sub(',',' ', string) # removing comma
    string=re.sub('_',' ', string) # removing underscore
    string=re.sub('-',' ', string) # remving hyphen
    string=re.sub('[^A-aZ-z ]',' ', string) # removing all special charcters  expect space  and 
    #if diffent language wordcome it will also take care of it.
    string=re.sub('\[',' ', string) # removing square brackets
    string=re.sub('\]',' ', string) # 
    string=re.sub(' {2,}',' ', string) # replacing two or more spaces with one space
    string=string.strip()
    # removing stop words from text
    # https://gist.github.com/sebleier/554280
    string = ' '.join(e for e in string.split() if e.lower() not in stopwords)
    string=string.lower()
    string = re.sub(r'\b\w{1,2}\b'," ",string) #remove words <2
    string=re.sub(' {2,}','', string) # replacing two or more spaces with one space
    return string   
  

  ################### getting preprocessing done #####################################

  #Creating new columns for preprocessed text, title and author name.
  from tqdm import tqdm
  def get_preprocess(column):
      pre_pro_text=[]
      for i in tqdm(column):
          pre_text=''
          pre_text=preprocessig_text(i)
          pre_pro_text.append(pre_text)
      return pre_pro_text

  ######################################################################################

  pre_text_test=get_preprocess(csv_text)
  pre_title_test=get_preprocess(csv_title)
  pre_author_test=get_preprocess(csv_author)
  print('Preprocessing done')

  #creating dictnory of all preprocessed list.
  # df_dic={"pre_text": pre_text_test,
  #         "pre_title":pre_title_test,
  #         "pre_author":pre_author_test}

  # # creating data frame
  # df=pd.DataFrame(df_dict,coloums=['pre_text','pre_title','Pre_author'])
  # csv=df

  csv['pre_text']=pre_text_test
  csv['pre_title']=pre_title_test
  csv['pre_author']=pre_author_test

  ############### Merging all data into single data ##############################

  csv['merged']= csv['pre_text'] + csv['pre_title'] + csv['pre_author']

  ####################### Tokenization of data ##################################

  encode_text_test=token_glove.texts_to_sequences(csv['merged'])

  ########################### padding of data ##################################

  max_length=int(1014)  # 95 precentile of the words in X_train.

  encode_text_test_paded=tf.keras.preprocessing.sequence.pad_sequences(encode_text_test, maxlen=max_length, padding='post')


  #################################### Loading best model for prediction   #################################


  def predict_test_thre(proba, threshould):
    predictions = []
    for i in proba:
        if i>=0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

  def Test_prediction(model,x_tr):
    y_predict_tr=model.predict(x_tr)
    pred=predict_test_thre(y_predict_tr, threshould=0.5)
    return pred
  print('Prediction is in process')
  # loading model

  lstm_cnn_predictions=Test_prediction(lstm_cnn_model,encode_text_test_paded)
  csv['Prediction']=lstm_cnn_predictions
  sub_NN=csv['Prediction']
   
  print('prediction done')

  return sub_NN
  
  


#########################################################################

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index1')
def index():
    return flask.render_template('index.html')


@app.route('/index', methods=['POST', 'GET'])
def predict():
    print ("Inside predict, ", request.method)
    print ("Request {}: ", request)

    if request.method=='POST':
        csv_file=request.files.get('csv_file')
        csv = pd.read_csv(csv_file)


        # print(path_in)
        df=final_prediction(csv)
        count=0
        if int(df)==1:
            string='Unreliable news'
        
        else:
            string='Reliable news'


        return string

        


    return flask.render_template('index.html')

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
