#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from datetime import datetime
import time
from statistics import mean,stdev
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
# # Classification with Multipler classifiers

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier

#from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from dateutil import parser
import json

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)


def count_from_str(s):
    if str(s) == "nan":
        return 0 
    else:
        return len(s.split("|"))

def videos_per_day(l):
    l = sorted(l) #ascending
    max_dt = datetime.fromtimestamp(l[-1])
    min_dt = datetime.fromtimestamp(l[0])
    return (max_dt-min_dt).days/len(l)


#days between videos
def interposting_time(l):
    if len(l) == 1:
        return [0,0]
    days_diff = []
    l = sorted(l)
    for i in range(0,len(l)-1):
        d0 = datetime.fromtimestamp(l[i])
        d1 = datetime.fromtimestamp(l[i+1])
        diff = (d1-d0).days
        days_diff.append(diff)
    #print(len(days_diff))
    if len(days_diff) == 1:
        return [mean(days_diff), 0]
    return [mean(days_diff), stdev(days_diff)]


def percentage_musicOriginal(l):
    return sum(l)/len(l)


def create_10bins(bmin,bmax):    
    return np.linspace(bmin,bmax,11)

def create_bins(arr1):
    
    # finding the 1st quartile
    q1 = np.quantile(arr1, 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(arr1, 0.75)
    med = np.median(arr1)

    # finding the iqr region
    iqr = q3-q1

    # finding upper and lower whiskers
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    bins = [lower_bound, q1, med, q3, upper_bound]
    return(bins)

def count_in_bins(l, bins, nbins):
    #return occurrences in 6 bins, first is below min, last is above max
    bincount = [0]*nbins
    n_el = 0
    for el in l:
        if str(el) == "nan" or str(el) == "ERROR" or str(el) == "NONE":
            continue
        else:
            n_el += 1
        found = False
        for i,b in enumerate(bins):
            if float(el) < float(b):
                bincount[i] += 1
                found = True
                break 
        if not found:
            bincount[-1] += 1
    bincount = [b/n_el if n_el> 0 else 0 for b in bincount]
    return bincount

def countOccurrence(l,val):
    count = 0
    n_el = 0
    for n in l:
        if str(n) == "nan":
            continue
        else:
            n_el += 1
        if n == val:
            count += 1
    if n_el == 0:
        return 0
    return count/n_el


class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 


class FeatureBinsCalculator(BaseEstimator, TransformerMixin):

    def __init__(self, duration_processing = False, general_processing = False):
        self.duration_processing = duration_processing
        self.general_processing = general_processing
        self.like_bins = None
        #self.share_bins = None
        self.comment_bins = None
        self.view_bins = None
        self.duration_bins = None
        self.k = None
        self.features_names = None
        
    def fit(self, X, y=None):
        if self.duration_processing == True:
            if self.k is None:
                self.k = df_v_processed.groupby('authorId').agg(list).reset_index().set_index('authorId')
            self.duration_bins = create_bins(df_v_processed[df_v_processed['authorId'].isin(list(X.index))]['duration'].values)
        if self.general_processing == True:
            if self.k is None:
                self.k = df_v_processed.groupby('authorId').agg(list).reset_index().set_index('authorId') 
            self.like_bins = create_bins(df_v_processed[df_v_processed['authorId'].isin(list(X.index))]['likes'].values)
            #self.share_bins = create_bins(df_v_processed[df_v_processed['authorId'].isin(list(X.index))]['shares'].values)
            self.comment_bins = create_bins(df_v_processed[df_v_processed['authorId'].isin(list(X.index))]['comments'].values)
            self.view_bins = create_bins(df_v_processed[df_v_processed['authorId'].isin(list(X.index))]['views'].values)
        return self
        #return X_

    def transform(self, X, y = None):
        X_ = X.copy()
        if self.duration_processing == True:
            X_['duration_distr'] = X_.apply(lambda row: count_in_bins(self.k.loc[row.name]['duration'], self.duration_bins, 6), axis= 1)
        if self.general_processing == True:    
            X_['likes_distr'] = X_.apply(lambda row: count_in_bins(self.k.loc[row.name]['likes'], self.like_bins, 6), axis= 1)
            #X_['shares_distr'] = X_.apply(lambda row: count_in_bins(self.k.loc[row.name]['shares'], self.share_bins, 6), axis= 1)
            X_['comments_distr'] = X_.apply(lambda row: count_in_bins(self.k.loc[row.name]['comments'], self.comment_bins, 6), axis= 1)
            X_['views_distr'] = X_.apply(lambda row: count_in_bins(self.k.loc[row.name]['views'], self.view_bins, 6), axis= 1)
            
        
        col_to_unpack = []
        col_to_drop = []
        if self.duration_processing == True:
            col_to_unpack += ['duration']
            col_to_drop += ['duration_distr']
        if self.general_processing == True:
            col_to_unpack += ["likes","comments","views"]
            col_to_drop += ['likes_distr', 'comments_distr','views_distr']

        # "unpack distributions into features"
        for x in col_to_unpack:
            for i in range(0,6):
                X_[f"{x}_bin_{i}"] = X_.apply(lambda row: row[f'{x}_distr'][i], axis = 1)
        
        

        X_.drop(columns=col_to_drop, inplace=True)
        
        self.features_names = list(X_.columns)
        #print(list(X_.columns))
        return X_

class columnKeeperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        #print(self.columns)
        return X[self.columns]

    def fit(self, X, y=None):
        return self 


def get_results(key, result):
    precision = result[key]['precision']
    recall = result[key]['recall']
    f1_score = result[key]['f1-score']
    accuracy = result['accuracy']
    
    key = '-'.join(key.split(' '))
    
    return [key, precision, recall, f1_score, accuracy]

def to_timestamp(s):
    date = parser.parse(s)
    return int(date.timestamp())

def remove_errors_nones(s):
    if str(s) == "ERROR" or str(s) == "NONE" or str(s) == "nan":
        return np.nan
    else:
        return int(float(s))

###############PREPROCESS DATASET###############


print("preprocessing dataset...")

df_videos = pd.read_csv("../Dataset/youtube_df_videos_FINAL_BEFORE_REMOVING.csv")
df_infl = pd.read_csv("../Dataset/youtube_final_influencers_FINAL.csv")
df_infl.rename(columns={"tier": "actual_tier"}, inplace = True)
df_infl.rename(columns={"handle": "id"}, inplace = True)
df_videos.rename(columns={"author": "authorId"}, inplace = True)
print(df_videos.columns)
df_videos['createTime'] = df_videos.apply(lambda row: to_timestamp(row['timestamp']), axis =1)

df_videos['likes'] = df_videos.apply(lambda row: remove_errors_nones(row['likes']), axis = 1)
#df_videos['likes'] = df_videos['likes'].astype(int)

col_to_keep = ['authorId','likes','views','comments','createTime','video_hasText','duration','height','width','heightXwidth']


# col_to_keep = ['createTime', 'isAd',
#        'isMuted', 'authorId', 
#        'authorContinent', 'musicOriginal', 
#        'height', 'width', 'heightXwidth', 'duration', 'definition', 
#        'mentions', 'hashtags', 'effects', 'authorTier', 
#        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
#        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
#        'time_signature', 'popularity', 'explicit', 'video_hasText']

col_to_keep_infl = ['id','continent','actual_tier','has_signature','has_bioLink','tot_videos','tot_views']


df_v_processed = df_videos[col_to_keep]

df_infl = df_infl[col_to_keep_infl]

day_map = {
    'Monday' : 0, 
    'Tuesday' : 1,
    'Wednesday': 2, 
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
df_v_processed['weekday'] = df_v_processed.apply(lambda row: day_map[time.strftime('%A',time.localtime(row['createTime']))], axis = 1 )

print(df_infl.columns)


one_hot = pd.get_dummies(df_infl['continent'])
df_infl = df_infl.drop('continent', axis = 1)
df_infl = df_infl.join(one_hot)

#calculate weekdays distribution
k = df_v_processed.groupby('authorId').agg(list).reset_index().set_index('authorId')
df_infl['weekdays_distr'] = df_infl.apply(lambda row: list(np.histogram(k.loc[row['id']]['weekday'], bins = list(range(8)) ,density= True)[0]), axis= 1)

# unpack weekdays distribution
for i in range(0,7):
    df_infl[f"weekday_{i}"] = df_infl.apply(lambda row: row[f'weekdays_distr'][i], axis = 1)

df_infl['videos_per_day']  = df_infl.apply(lambda row: videos_per_day(k.loc[row['id']]['createTime']), axis=1)


#inter-posting time
lists = list(k.iloc[0].createTime)
df_infl['interposttime_mean'] = df_infl.apply(lambda row: interposting_time(k.loc[row['id']]['createTime'])[0], axis=1)
df_infl['interposttime_std'] = df_infl.apply(lambda row: interposting_time(k.loc[row['id']]['createTime'])[1], axis=1)

#calculate aggregate features
#df_v_processed['definition'] = df_v_processed.apply(lambda row: int(row['definition'].replace("p","")), axis=1)


print("join others...")

#join other audio, video, caption emoticon features
df_to_join = ['ADV_user_features.csv','AU_user_features.csv','demog_user_features.csv','emot_user_features.csv','RGB_user_features.csv', 'youtube_CapEmoticon.csv']
df_to_join = ["../Dataset/"+d for d in df_to_join]


good_influencers = []

for df in df_to_join:
    print("joining ", df)
    temp_df = pd.read_csv(df, index_col='AuthorId')
    df_infl = df_infl.join(temp_df, on='id',rsuffix = df.split("/")[-1])

    if "RGB" in df:
        good_influencers = temp_df.index.values

print(good_influencers)
    #df_infl.drop(columns=[df.split("/")[-1]])
#print(df_infl.head())


#df_v_processed = df_v_processed[df_v_processed['authorId'].isin(good_influencers)]

agg_metrics = ['min','max','mean','std']
features_to_agg = ['height', 'width','heightXwidth', 'duration','likes','views','comments']

for f_agg in agg_metrics:
    
    k = df_v_processed.groupby('authorId')[features_to_agg].agg(f_agg).reset_index().set_index('authorId')
    for f in features_to_agg:
        df_infl[f"{f}_{f_agg}"] = df_infl.apply(lambda row: k.loc[row['id']][f], axis= 1)
    


#calculate has_text
k = df_v_processed.groupby('authorId').agg(list).reset_index().set_index('authorId')
df_infl['video_hasText_perc'] = df_infl.apply(lambda row: percentage_musicOriginal(k.loc[row['id']]['video_hasText']), axis=1)


def select_best_model(df_ab, tier, feature, iter):

    clf = LogisticRegression(random_state=123,max_iter=10000)
    # clf = RandomForestClassifier(random_state=123, n_jobs=16)
    couple = tier[0]+"-"+tier[1]
    df_t = (df_ab[(df_ab['columns']==feature) & (df_ab['iter']==iter) & (df_ab['couples']==couple) & (df_ab['classifier']=='LR')]).reset_index(drop=True)
    data = df_t.iloc[0]['params']
    print()

    # print('data:',data)
    data = data.replace("\'", "\"")
    data = data.replace("None", "null")
    best_params = json.loads(data)
    print(best_params)

    # setting best parameters to classifiers
    for k,v in best_params.items():
        nKey = k.split('clf__')[1]
        setattr(clf, nKey, v)

    return clf
###############PREDICTIONS###############


#pandas results
df_results = pd.DataFrame(columns = ['couples', 'classifier', 'iter', 'precision', 'recall','f1','accuracy', 'params', 'k_best','columns'])

df_feature_imp = None


couples =  [('nano','micro'), ('micro','mid'), ('mid','macro'), ('macro','mega')]
algorithms = ["LR"] #["XGB", "RF", "LR","DT","NB","SVM","DNN","KNN"] #['RF',"XGB","LR","DT","NB","SVM","MLP","KNN"]
k_bests = ['all']

vid_features = ['height_min', 'width_min', 'heightXwidth_min',
                'height_max', 'width_max', 'heightXwidth_max', 
                'height_mean', 'width_mean', 'heightXwidth_mean',  
                'height_std', 'width_std', 'heightXwidth_std' ]

for c in vid_features:
    df_infl.rename(columns={c: 'video_'+c},inplace=True)



col_to_drop = ['weekdays_distr']


video_features = []
text_features = []
audio_features = []
general_features = []

print(list(df_infl.columns))

for c in df_infl.columns:
    if c in col_to_drop:
        continue 
    elif c == "tot_videos" or c == "videos_per_day" or c=="tot_views":
        general_features.append(c)
    elif "audio_" in c or "spotify" in c:
        audio_features.append(c)
    elif "video_" in c or "duration" in c:
        video_features.append(c)
    elif "text_"in c:
        text_features.append(c)
    else:
        if c != "actual_tier" and c != "id":
            general_features.append(c)

all = general_features + audio_features + video_features + text_features
print("TOT FEATURES LEN:",len(video_features)+len(audio_features)+len(text_features)+len(general_features))

print("GENERAL:::::")
for c in general_features:
    print(c)


#df_infl.to_csv("DF_INFL.csv", index=False)
#print(list(df_infl.columns))
#print("saved!")


from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def flatten(l):
    return [item for sublist in l for item in sublist]

column_combinations = [general_features, video_features, audio_features, text_features, all] #[ video_features,general_features, audio_features, text_features]

# column_combinations = list(powerset(column_combinations))
column_names = ["('general',)", "('video',)", "('audio',)", "('text',)", "('video', 'general', 'audio', 'text')"]

file_save_name = ['general','video','audio','text','all']
# column_names_combinations = ['general','all']

# column_names_combinations = ["video","general","audio","text"]
# column_names_combinations = list(powerset(column_names_combinations))

#REMOVE INFLUENCERS THAT ARE NOT GOOD
df_infl = df_infl[df_infl['id'].isin(good_influencers)]
print("total influencers: ",len(df_infl))

df_infl.to_csv("DF_INFL.csv",index=False)

df_param = pd.read_csv('Youtube_all_for_FI.csv')

for i, col_ablation in enumerate(column_combinations):
    df_feature_imp = None
    for alg in algorithms:
        for c in couples:
            if len(col_ablation) == 0:
                continue
            # col_ablation = flatten(col_ablation)

            columns_used = column_names[i]
            # if 'video' not in columns_used:
            #     print(f"skipping {columns_used}")
            #     continue
            
            for k_best in k_bests:

                print(f"====== ANALYZING {c} - {alg} - {k_best} - {columns_used}======")

                df_ML = df_infl[(df_infl['actual_tier'] == c[0]) | (df_infl["actual_tier"] == c[1])].copy()

                print("influencers: ", len(df_ML))

                df_ML.drop(columns=col_to_drop, inplace=True)
                df_ML.fillna(0, inplace=True)

                y = df_ML['actual_tier']
                # le = LabelEncoder()
                # y = le.fit_transform(y)
                X = df_ML.drop(columns=['actual_tier'])

                sss = StratifiedKFold(n_splits=10, shuffle = True, random_state=0)

                outer_f1 = list()
                outer_acc = list()
                outer_rec = list()
                outer_prec = list()

                index_iter = 0
                for train_index,test_index in sss.split(X, y):

                    df_tr = X.iloc[train_index]
                    df_te = X.iloc[test_index]
                    
                    df_tr = df_tr.set_index('id')
                    df_te = df_te.set_index('id')
                    
                    le = LabelEncoder()
                    y_tr = y.iloc[train_index]
                    train_le = le.fit(y_tr)
                    y_tr = train_le.transform(y_tr)
                    y_test = y.iloc[test_index]
                    y_test = train_le.transform(y_test)
                        
                    print(f"Train shape: {df_tr.shape}, Test shape: {df_te.shape}")
                    
                    # cv_inner = StratifiedKFold(n_splits = 5, random_state = 123 , shuffle = True).split(df_tr, y_tr)


                    
                    # clf, param_grid = select_algorithm(alg)
                    try:
                        clf = select_best_model(df_ab=df_param, tier=c, feature=columns_used, iter=index_iter)
                    except:
                        continue

                    duration_process = False
                    general_process = False
                    if 'video' in columns_used:
                        duration_process = True
                    if 'general' in columns_used:
                        general_process = True

                    pipe = Pipeline(steps=[ 
                        ("column_select", columnKeeperTransformer(col_ablation)),
                        ("distr_calc", FeatureBinsCalculator(duration_processing=duration_process, general_processing=general_process)), 
                        ("zero_var", VarianceThreshold(0.00)), 
                        ('scale', StandardScaler()),
                        ("skb", SelectKBest(k = k_best)),  
                        ('clf',clf)
                    ])

                    # define search
                    # search = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=cv_inner, refit=True, n_jobs = 20)

                    # execute search
                    print("fitting...")
                    pipe.fit(df_tr, y_tr)
                    # get the best performing model fit on the whole training set
                    # best_model = result.best_estimator_
                    # evaluate model on the hold out dataset
                    yhat = pipe.predict(df_te)
                    # evaluate the model
                    f1_res = f1_score(y_test, yhat, average="macro")
                    acc = accuracy_score(y_test, yhat)
                    prec = precision_score(y_test, yhat, average="macro")
                    rec = recall_score(y_test, yhat, average="macro")

                    # print(f1_res)
                    # # feature importance
                    if df_feature_imp is None:
                        df_feature_imp = pd.DataFrame(columns = ["couple","classifier","kbest","iteration","f1", "columns"] + pipe['distr_calc'].features_names)
                    
                    print(list(pipe['distr_calc'].features_names))


                    try:
                        # sorted_idx = np.argsort(best_model['clf'].feature_importances_)[::-1]
                        used_features = np.array(pipe['distr_calc'].features_names)
                        used_features = used_features[pipe['zero_var'].get_support(indices=True)]
                        used_features = used_features[pipe['skb'].get_support(indices = True)]
                        
                        res_dict = {
                            "couple" : c,
                            'classifier': alg,
                            "kbest": k_best,
                            "iteration": index_iter,
                            "f1": f1_res,
                            "columns": columns_used
                        }
                        coef_ = pipe['clf'].coef_
                        coef_ = coef_.flatten()
                        for idx, feature in enumerate(used_features):
                            coef_val = coef_[idx]
                            res_dict[feature] = abs(coef_val)
                        

                        df_feature_imp.loc[len(df_feature_imp)] = res_dict
                        df_feature_imp.to_csv(f"Youtube_LR_{file_save_name[i]}_feature_importance.csv", index=False)


                        # sorted_features = used_features[sorted_idx]
                        # sorted_imp = pipe['clf'].feature_importances_[sorted_idx]
                        # for f, imp in zip(sorted_features, sorted_imp):
                        #     print(f,imp)
                    except:
                        print(f'{alg} does not have feature importance')


                    # report progress
                    # print('>f1=%.3f, est=%.3f, acc=%.3f, cfg=%s' % (f1_res, result.best_score_, acc, result.best_params_))

                    df_results.loc[len(df_results)] = {
                        'couples': c[0]+"-"+c[1], 
                        'classifier': alg, 
                        'iter': index_iter, 
                        'precision': prec,
                        'recall': rec,
                        'f1': f1_res,
                        'accuracy': acc, 
                        # 'params': result.best_params_,
                        'k_best': k_best,
                        'columns': columns_used
                    }

                    df_results.to_csv("Youtube_LR_fi_results_cross_val.csv", index = False)

                    index_iter += 1

                    




# dict_classifiers = {
                         # Naive_Bayes
# }



