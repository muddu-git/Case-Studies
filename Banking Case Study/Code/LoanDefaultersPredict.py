#Importing required libraries
import pandas as pd
import numpy as np
from fancyimpute import KNN
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score 
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("C:/Users/mudmoham/Documents/pr/Banking_Case_Study/output/cleaned_data.csv",parse_dates=["Account Creation Date","birth_number","loan grant date","card issued date"])

#Missing Value Analysis
pd.isnull(data).sum()

#Outlier Analysis
Num_Columns=[]
Cat_Columns=[]
for col in data.columns:
	if data[col].dtype==np.int64 or data[col].dtype==np.float64:
		Num_Columns.append(col)
	else:
		Cat_Columns.append(col)
		
con_data=data[Num_Columns]
cat_data=data[Cat_Columns]
for col in con_data.columns:
	q75,q25=np.percentile(con_data[col],[75,25])
	iqr=q75-q25
	minimum=round(q25-(iqr*1.5))
	maximum=round(q75+(iqr*1.5))
	con_data.loc[con_data[col]<minimum,col]=np.nan
	con_data.loc[con_data[col]>maximum,col]=np.nan

con_data=pd.DataFrame(KNN(k=3).complete(con_data),columns=con_data.columns)
bank_data=cat_data.join(con_data)

#Feature Selection
#Dropping Unnecessary features by observation
bank_data=bank_data.drop(["account_id","district_id_x","Account Creation Date","disp_id","client_id","birth_number","district_id","loan_id","loan grant date","card_id","card issued date","order_id","Recepient Account"],axis=1)

#Correalation Analysis
#Dropping Highly Correlated Varaibles
bank_data=bank_data.drop(["No of Municipalties","Total Credited Amount","Average Credit Balance"],axis=1)
#Chi-square test of Independence
for col in bank_data.columns:
	if 	(bank_data[col].dtype==np.object) & (col!="status"):
		chi2,p,dof,expected=chi2_contingency(pd.crosstab(bank_data["status"],bank_data[col]))
		print(col)
		print(p)
bank_data=bank_data.drop(["Recepient Bank"," order payment type","Gender","Disposition_Type","frequency"],axis=1)

for col in bank_data.columns:
	if bank_data[col].dtype==np.object:
		bank_data[col]=pd.Categorical(bank_data[col])
		bank_data[col]=bank_data[col].cat.codes	

#Feature Scaling
pre_processed_data=pd.DataFrame(MinMaxScaler().fit_transform(bank_data),columns=bank_data.columns)
pre_processed_data["status"]=round(pre_processed_data["status"],1)

#Dividing into train and test data sets
train=pre_processed_data[(pre_processed_data["status"]==0.) |(round(pre_processed_data["status"],2)==0.3)]
train["status"]=train["status"].replace({0.0:0,0.3:1})
xtrain=train.drop(["status"],axis=1)
ytrain=train["status"]
test=pre_processed_data[(round(pre_processed_data["status"],2)==0.7) |(pre_processed_data["status"]==1.)] 
xtest=test.drop(["status"],axis=1)
ytest=test["status"]

#Target class Imbalance Problem
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0,k_neighbors=3)
xtrain_imb,ytrain_imb=oversampler.fit_sample(xtrain,ytrain)
xtrain_imb=pd.DataFrame(xtrain_imb,columns=xtrain.columns)
ytrain_imb=pd.Series(ytrain_imb)

#Modelling
rf=RandomForestClassifier(random_state=0,max_depth=3,min_samples_leaf=15) 
cv_score=np.mean(cross_val_score(rf,xtrain_imb,ytrain_imb,cv=5))
print(cv_score)
rf.fit(xtrain_imb,ytrain_imb)
y_pred=rf.predict(xtest)
pd.Series(y_pred).value_counts()

#Writing Output
y_pred=pd.DataFrame(y_pred,index=xtest.index,columns=["Loan Defaulters"])
test_output=data[(data["status"]=="C") | (data["status"]=="D")]
test_output=test_output.join(y_pred)
output=test_output[round(test_output["Loan Defaulters"])==1]["account_id"].drop_duplicates()
output.to_csv("C:/Users/mudmoham/Documents/pr/Banking_Case_Study/output/loan_defaulter_accounts.csv",index=False,header="Account ID")







	
	
	
	





