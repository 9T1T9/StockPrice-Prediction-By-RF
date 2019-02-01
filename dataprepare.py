""" Label transactions with different patterns
Six patterns: Continuous Up, Sideways Up, Continuous
down, Sideways down, Flat, Unknown"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Count the number of clips in the area
def countin(highf,lowf,highl,lowl,close):
    growth=highl-highf
    PD=len(close)
    count=0
    for i in range(0,PD):
        highbound=highf+growth*i/(PD-1)
        lowbound=lowf+growth*i/(PD-1)
        if close[i]<=highbound and close[i]>=lowbound:
            count+=1
    return count

# Count the number of clips above the line
def countabove(lowf,lowl,close):
    count=0
    growth=lowl-lowf
    PD=len(close)
    for i in range(0,PD):
        lowbound=lowf+growth*i/(PD-1)
        if close[i]>=lowbound:
            count+=1
    return count

# Count the number of clips below the line
def countbelow(highf,highl,close):
    count=0
    growth=highl-highf
    PD=len(close)
    for i in range(0,PD):
        highbound=highf+growth*i/(PD-1)
        if close[i]<=highbound:
            count+=1
    return count

# Recognize continuous up pattern
def isCU(close,t,threshold):
    first=close[0]
    last=close[-1]
    growth=last-first
    Flag=False
    PD=len(close)
    highf=first+growth*t
    lowf=first-growth*t
    highl=last+growth*t
    lowl=last-growth*t
    cnt=countin(highf,lowf,highl,lowl,close)
    if cnt/PD>=threshold:
        Flag=True
    return Flag
        
# Recognize sideways up pattern
def isSU(close,a1,a2,t,threshold):
    first=close[0]
    last=close[-1]
    PD=len(close)
    growth=last-first
    Flag=False
    for i in range(round(0.5*PD),round(0.8*PD)):
        if close[i]-first<=growth*a2 and close[i]-first>=growth*a1:
            highl=first+growth*a2
            lowl=first+growth*a1
            highf= first+(highl-close[i])
            lowf=first-(close[i]-lowl)
            cntin=countin(highf,lowf,highl,lowl,close[0:i+1])
            lowlast=last-growth*t
            cntabove=countabove(lowl,lowlast,close[i:PD])
            if(cntin/(i+1))>=threshold and (cntabove/(PD-i))>=threshold:
                Flag=True
                return Flag
    return Flag

# Recognize continuous down pattern
def isCD(close,t,threshold):
    first=close[0]
    last=close[-1]
    # Negative
    growth=last-first
    PD=len(close)
    highf=first-growth*t
    lowf=first+growth*t
    highl=last-growth*t
    lowl=last+growth*t
    Flag=False
    cnt=countin(highf,lowf,highl,lowl,close)
    if cnt/PD>=threshold:
        Flag=True
    return Flag

# Recognize slides down pattern
def isSD(close,a1,a2,t,threshold):
    first=close[0]
    last=close[-1]
    PD=len(close)
    # Negative
    growth=last-first
    Flag=False
    for i in range(round(0.5*PD),round(0.8*PD)):
        if close[i]-first<=growth*a1 and close[i]-first>=growth*a2:
            highl=first+growth*a1
            lowl=first+growth*a2
            highf= first+(highl-close[i])
            lowf=first-(close[i]-lowl)
            cntin=countin(highf,lowf,highl,lowl,close[0:i+1])
            highlast=last-growth*t
            cntbelow=countbelow(highl,highlast,close[i:PD])
            if(cntin/(i+1))>=threshold and (cntbelow/(PD-i))>=threshold:
                Flag=True
                return Flag
    return Flag

# Recognize flat pattern
def isFL(close,t,threshold):
    first=close[0]
    last=close[-1]
    Flag=False
    PD=len(close)
    growth=last-first
    highf=first+abs(growth)*t
    lowf=first-abs(growth)*t
    highl=last+abs(growth)*t
    lowl=last-abs(growth)*t
    cnt=countin(highf,lowf,highl,lowl,close)
    if(cnt/(PD-1))>=threshold:
        Flag=True
    return Flag

# Label the prediction duration
def labeling(close,a1,a2,t,G,threshold):
    growth=(close[-1]-close[0])/close[0]
    label=5
    if growth>=G:
        if isCU(close,t,threshold):
            label=1
        if isSU(close,a1,a2,t,threshold):
            label=2
    if growth<=-G:
        if isCD(close,t,threshold):
            label=3
        if isSD(close,a1,a2,t,threshold):
            label=4
    if growth>-G and growth<G:
        if isFL(close,t,threshold):
            label=0
    return label    
    
def slideforest(close,PD,a1,a2,t,G,threshold):
    time=list(np.arange(0,80,4))
    label=[]
    for i in range(0,len(close)-PD+1):
        sample=close[i:i+PD]
        haha=labeling(sample,a1,a2,t,G,threshold)
        label.append(labeling(sample,a1,a2,t,G,threshold))
        
        if(haha==0):
            plt.figure()
            plt.plot(time,sample)
            plt.xlabel('time')
            plt.ylabel('price')
#            plt.xticks(np.arange(0,300,15))
            plt.yticks(np.arange(min(sample)-7,max(sample)+7,3))
            if haha==0:
                plt.suptitle('Flat')
            if haha==1:
                plt.suptitle('Continuous Up')
            if haha==2:
                plt.suptitle('Slides Up')
            if haha==3:
                plt.suptitle('Continuous Down')
            if haha==4:
                plt.suptitle('Slides Down')
            if haha==5:
                plt.suptitle('Unknown')
            plt.show()
        
def generate_labels(df,PD,a1,a2,t,G,threshold):
    close=df['denoised_data'].values.tolist()
    label=[]
    for i in range(0,len(close)-PD+1):
        sample=close[i:i+PD]
        label.append(labeling(sample,a1,a2,t,G,threshold))
    return label

def convert(close,period):
    window=[]
    for i in range(0,len(close)):
        if (i%period)==0:
            window.append(close[i])
    return window

def generate_train(df,PD,MD,feature):
     features=df[feature].set_index('time').T.to_dict('list')
     training=[]
     for i in range(0,len(df)-PD+1):
         a_train={}
         for time in range(i,i+MD):
             a_train[time]=features[time]
         training.append(a_train)
     return training
 
def generate_test(df,PD,TD,feature):
    features=df[feature].set_index('time').T.to_dict('list')
    test=[]
    for i in range(0,len(df)-PD+1):
        a_test={}
        for time in range(i,i+TD):
            a_test[time]=features[time]
        test.append(a_test)
    return test
 
if __name__ =='__main__':
    df=pd.read_csv('/home/zyt/bdt/5001/project/complete_data_no_label.csv', engine='python')
    close=df['denoised_data'].values.tolist()
    slideforest(close,20,0.05,0.4,0.5,0.02,0.95)
#    close=convert(close,1)
#    training,labels= slide(close,20,0.05,0.4,0.5,0.02,0.95)
#    train=generate_train(df,20,15)
#    test=generate_test(df,20,10)
#    label=generate_labels(df,20,0.05,0.4,0.5,0.02,0.95)
#    labels.remove(labels[0])
#    df_length=len(df)
#    nanlist=[np.nan for i in range(df_length-len(labels))]
#    new_labels=labels+nanlist
#    label_num=[new_labels.count('CU'),new_labels.count('SU'),new_labels.count('CD'),new_labels.count('SD'),new_labels.count('FL'),new_labels.count('UK')]
#    df['label']=new_labels
#    df.to_csv('label_data.csv',index=False)
    