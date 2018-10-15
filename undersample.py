import warnings
warnings.filterwarnings('ignore')



import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # to split the data
from sklearn.cross_validation import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report


data = pd.read_csv("train.csv",header = 0)
data.info()

# sns.countplot("category",data=data)


# Count_two_four = len(data[data["category"]==2])+ len(data[data["category"]==4])
# Count_normal = len(data)-Count_two_four
# print("\nOther data = "+str(Count_normal))
# print("\nTwo four data = "+str(Count_two_four))

def getUndersampledData():
	#most infrequent dataset
	Count_one = len(data[data["category"]==1])
	sample_one=np.array((data[data.category==1].index))
	undersample_zero = np.array(np.random.choice(np.array((data[data.category==0].index)),Count_one,replace=False))
	undersample_two = np.array(np.random.choice(np.array((data[data.category==2].index)),Count_one,replace=False))
	undersample_three = np.array(np.random.choice(np.array((data[data.category==3].index)),Count_one,replace=False))
	undersample_four = np.array(np.random.choice(np.array((data[data.category==4].index)),Count_one,replace=False))
	undersample_data=np.concatenate([undersample_zero,sample_one,undersample_two,undersample_three, undersample_four])
	undersample_data = data.iloc[undersample_data,:]
	return undersample_data

files = 4 #Because the data is roughly four times in size of the given data
filenames=["undersample_data"+str(i+1)+".csv" for i in range(files)]
for i in range(files):
	filename=filenames[i]
	undersample_data=getUndersampledData()
	undersample_data.to_csv(filename,index=False)
	# bigger_undersample_data = np.concatenate([bigger_undersample_data,undersample_data])
	undersample_data.info()
	sns.countplot("category",data=undersample_data)

combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )
combined_csv.to_csv("combined_csv.csv", index=False)
plt.show()

#To merge


# two_four_indices= np.array(list(data[data.category==2].index)+ list(data[data.category==4].index))
# normal_indices = np.array([i for i in range(len(data)) if i not in two_four_indices])
# def undersample(normal_indices,two_four_indices,times):
# 	two_four_indices_undersample=np.array(np.random.choice(two_four_indices,(3*Count_normal)//5,replace=False))
# 	undersample_data=np.concatenate([normal_indices,two_four_indices_undersample])
# 	undersample_data = data.iloc[undersample_data,:]
# 	print("the 2-4 data proportion is :",(len(undersample_data[undersample_data.category==2])+len(undersample_data[undersample_data.category==4]))/len(undersample_data))
# 	return undersample_data
# undersample_data = undersample(normal_indices,two_four_indices,1)#len(two_four_indices)/len(normal_indices))
