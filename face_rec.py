import numpy as np
import pandas as pd
import cv2

import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import os

import time
from datetime import datetime

hostname = 'redis-12675.c89.us-east-1-3.ec2.redns.redis-cloud.com'
portnumber = 12675
password = '6nkM2Z5Q9gXV4lsqiLuZ0UxNi5rsqNrV'
r = redis.StrictRedis(host=hostname,port= portnumber,password=password)
def retrive_data(name):
    retrive_dict= r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df =  retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role','facial_features']
    retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name','Role','facial_features']]


#configure face analysis
faceapp = FaceAnalysis(name = 'buffalo_sc',
                     root= 'insightface_model',
                     providers=[ 'CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640), det_thresh=0.3)

#ML search algo
def ml_search_algorithms(df,feature_column,test_vector,name_role = ['Name','Role'],thresh=0.5):
   df = df.copy()
   X_list = df[feature_column].tolist()
   x = np.asarray(X_list)
   similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
   similar_arr = np.array(similar).flatten()
   df['cosine'] = similar_arr

   data_filter = df.query(f'cosine>={thresh}')
   if len(data_filter)>0:
      data_filter.reset_index(drop=True,inplace=True)
      argmax = data_filter['cosine'].argmax()
      person_name, person_role = data_filter.loc[argmax][name_role]
   else:
      person_name = 'Unknown'
      person_role = 'Unknown'
   return person_name,person_role

def face_prediction(test_image,df,feature_column,name_role = ['Name','Role'],thresh=0.5):
   current_time = str(datetime.now())
   results = faceapp.get(test_image)
   test_copy = test_image.copy()
   for res in results:
      x1, y1, x2, y2 = res['bbox'].astype(int)
      embeddings = res['embedding']
      person_name, person_rule = ml_search_algorithms(df,feature_column,test_vector=embeddings,name_role=name_role,thresh=thresh)
      if person_name == 'Unknown':
         color = (0,0,255)
      else:
         color = (0,255,0)
      cv2.rectangle(test_copy,(x1,y1),(x2,y2),(0,255,0))
      text_gen = person_name
      cv2.putText(test_copy, text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,1)
      cv2.putText(test_copy, current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,1)
   return test_copy

class RealTimePred:
   def __init__(self):
      self.logs = dict(name=[], role=[], current_time=[])

   def reset_dict(self):
      self.logs = dict(name=[], role=[], current_time=[])

   def saveLogs_redis(self):
      dataframe = pd.DataFrame(self.logs)
      dataframe.drop_duplicates('name', inplace=True)
      name_list = dataframe['name'].tolist()
      role_list = dataframe['role'].tolist()
      ctime_list = dataframe['current_time'].tolist()
      encoded_data = []
      for name, role, ctime in zip(name_list, role_list, ctime_list):
         if name != 'Unknown':
            concat_string = f"{name}@{role}@{ctime}"
            encoded_data.append(concat_string)

         if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)

         self.reset_dict()

   def face_prediction(self, test_image, df, feature_column, name_role=['Name', 'Role'], thresh=0.5):
      current_time = str(datetime.now())
      results = faceapp.get(test_image)
      test_copy = test_image.copy()
      for res in results:
         x1, y1, x2, y2 = res['bbox'].astype(int)
         embeddings = res['embedding']
         person_name, person_role = ml_search_algorithms(df, feature_column, test_vector=embeddings,name_role=name_role, thresh=thresh)
         if person_name == 'Unknown':
            color = (0, 0, 255)
         else:
            color = (0, 255, 0)
         cv2.rectangle(test_copy, (x1, y1), (x2, y2), (0, 255, 0))
         text_gen = person_name
         cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
         cv2.putText(test_copy, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
         self.logs['name'].append(person_name)
         self.logs['role'].append(person_role)
         self.logs['current_time'].append(current_time)
      return test_copy


class RegistrationForm:
   def __init__(self):
      self.sample = 0
   def reset(self):
      self.sample = 0
        
   def get_embedding(self,frame):

      results = faceapp.get(frame,max_num=1)
      embeddings = None
      for res in results:
         self.sample += 1
         x1, y1, x2, y2 = res['bbox'].astype(int)
         cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
         text = f"samples = {self.sample}"
         cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
         embeddings = res['embedding']
            
      return frame, embeddings
    
   def save_data_in_redis_db(self,name,role):
      if name is not None:
         if name.strip() != '':
            key = f'{name}@{role}'
         else:
            return 'name_false'
      else:
         return 'name_false'
        
      if 'face_embedding.txt' not in os.listdir():
         return 'file_false'
        
        
      x_array = np.loadtxt('face_embedding.txt',dtype=np.float32)          
        
      received_samples = int(x_array.size/512)
      x_array = x_array.reshape(received_samples,512)
      x_array = np.asarray(x_array)       
        
      x_mean = x_array.mean(axis=0)
      x_mean = x_mean.astype(np.float32)
      x_mean_bytes = x_mean.tobytes()
        

      r.hset(name='academy:register',key=key,value=x_mean_bytes)

      os.remove('face_embedding.txt')
      self.reset()
      return True
