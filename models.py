import dataprepare
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import random

"""Construct random forest model class."""
class RF:
    def __init__(self, trainx,trainy,testx,testy):
        self.clf=RandomForestClassifier(class_weight="balanced")
        self.clf.fit(trainx,trainy)
        self.acc=self.evaluate(testx,testy)
        print("RF: ",self.evaluate(testx,testy))
        
    """Evaluate accuracy"""
    def evaluate(self,testx,testy):
        prediction=self.predict(testx)
        acc=accuracy_score(testy,prediction)
        return acc
    
    """Prediction"""
    def predict(self,feature):
        return self.clf.predict(feature).flatten()

"""Compress multiple dimensional features into one dimension."""
def featurecompression(raw):
    features=[]
    for i in raw:
        time_feature=[feature for t in i.values() for feature in t]
        features.append(time_feature)
    return np.array(features)

"""Use interpolation method to expand corresponding time of features."""    
def expandtime(raw,target):
    new_raw=[]
    for data in raw:
        x=list(data.keys())
        x_new=np.linspace(x[0],x[-1],num=target)
        features=np.array(list(data.values()))
        new_features=np.zeros([target,features.shape[1]])
        for i in range(0,features.shape[1]):
            y=features[:,i]
            f = interp1d(x, y)
            new_features[:,i]=f(x_new)
        new_data=dict(zip(list(range(0,target)),[list(new_features[i,:]) for i in range(0,new_features.shape[0])]))
        new_raw.append(new_data)
    return new_raw

"""Calculate distance between ith row and jth row of train data"""
def dis(i,j,train):
    i=np.array(train[i])
    j=np.array(train[j])
#    print(i)
#    print(j)
    return sum((i-j)**2)
def dist(i,train):
    dis=sum([sum((np.array(train[i])-np.array(train[j]))**2) for j in range(0,len(train)) if j!= i])
    return dis
        
"""Sampling data to make the number of instances for each classes same."""
def undersampling(train,labels):
    label_name=[0,1,2,3,4,5]
    labels_dis=[labels.count(i) for i in label_name]
#    print(labels_dis)
    min_samples=min(labels_dis)
    min_label=labels_dis.index(min_samples)
    #label_name.remove(min_label)
    sample_train=[]
#    sample_test=[]
    sample_labels=[]
    for label in label_name:
        indices=[i for i in range(len(labels)) if labels[i]==label]
        ntrain=[train[i] for i in indices]
#        ntest=[test[i] for i in indices]
#        dual_distances=[[dis(i,j,ntrain) for i in range(0,len(ntrain))] for j in range(0,len(ntrain))]
#        print(dual_distances)
#        distances=[(i,dist(i,ntrain)) for i in range(0,len(ntrain))]
#        print(distances)
#        distances.sort(key=lambda x:x[1])
#        print(distances)
        if label!=min_label:
            sample_indices=random.sample(range(0,len(ntrain)),min_samples)
        else:
            sample_indices=[i for i in range(0,len(ntrain))]
        sample_train+=[ntrain[i] for i in sample_indices]
#        sample_test+=[ntest[i] for i in sample_indices]
        sample_labels+=[label for i in sample_indices]
    return sample_train,sample_labels
    
"""Select best features combiations by Forward Sequential Search."""
def feature_selection(df,PD,MD):
    """Get all features except time"""
    df.drop(['VR'],axis=1,inplace=True)
    features=list(df.columns)
    """"The price is an important feature and can always stay in the test feature list."""
    test_feature=[features[0],features[26],features[27],features[28],features[29]]
    """Get labels."""
    labels=dataprepare.generate_labels(df,PD,0.05,0.4,0.5,0.01,0.95)
    """Get accuracy of model with denoised data first."""
    new_df=featurecompression(dataprepare.generate_train(df,PD,MD,test_feature))
    """Undersampling."""
    new_df,labels=undersampling(new_df,labels)
    new_df=np.array(new_df)
    labels=np.array(labels)
    new_df=new_df.astype(np.float32)
    labels=labels.astype(np.float32)
    trainx,testx,trainy,testy=train_test_split(new_df, labels, test_size=0.3, random_state=66,shuffle=True)
#    test=featurecompression(expandtime(dataprepare.generate_test(df,20,10,test_feature),15))
    print(test_feature)
    """Train RF model."""
    model=RF(trainx,trainy,testx,testy)
    """Construct a dictionary to store accuracy of models for using different number of features."""
    Accuracy={str(test_feature):model.acc}
    #return model.acc
    best_model=model
    model_candidates=[best_model]
    """Generate accuracy for models applied on different features."""
    while len(test_feature)!=len(features):
        """Initialize accuracy and feature list."""
        acc=0
        local_best_feature=[]
        local_best_model=best_model
        local_test_feature=test_feature.copy()
        for fea in features:
            if fea not in local_test_feature:
                local_test_feature.append(fea)
                print(local_test_feature)
                """Generate train and test data with different features."""
                labels=dataprepare.generate_labels(df,PD,0.05,0.4,0.5,0.01,0.95)
                new_df=featurecompression(dataprepare.generate_train(df,PD,MD,local_test_feature))
                new_df,labels=undersampling(new_df,labels)
                new_df=np.array(new_df)
                labels=np.array(labels)
                new_df=new_df.astype(np.float32)
                labels=labels.astype(np.float32)
                trainx,testx,trainy,testy=train_test_split(new_df, labels, test_size=0.3, random_state=69)
                """Train RF model."""
                model=RF(trainx,trainy,testx,testy)
                """Get best features when number of features is limited."""
                if acc<model.acc:
                    local_best_feature=local_test_feature.copy()
                    acc=model.acc
                    local_best_model=model
                local_test_feature.remove(fea)
        """Store local best feature and acuracy in the dictionary.
        Store local best model in the list."""
        Accuracy[str(local_best_feature)]=acc
        model_candidates.append(local_best_model)
        """Update test_feature"""
        test_feature=local_best_feature.copy()
        
    best_feature=max(Accuracy,key=Accuracy.get)
    best_model=model_candidates[len(best_feature.split(','))-5]
    return best_feature,best_model,Accuracy[str(best_feature)]

"""Main"""
if __name__=='__main__':
    df=pd.read_csv('/home/zyt/bdt/5001/project/complete_data_no_label.csv')
    feature_selection(df,30,10)
#    MD=list(range(1,20))
#    acc=[]
#    for i in MD:
#        acc.append(feature_selection(df,20,i))
#    plt.plot(MD,acc)
    