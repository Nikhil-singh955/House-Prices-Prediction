import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, GradientBoostingRegressor,VotingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso,LinearRegression
from sklearn.metrics import mean_squared_error
################ importing Data ###################

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape
test.shape

train_y = train['SalePrice']
train_x = train.drop(['Id','SalePrice'],axis = 1)
train_x.shape

test_x = test.drop(['Id'],axis = 1)
test_x.shape

#### categorical and numerical features #######
categorical_features = train_x.select_dtypes(include = 'object').columns
numerical_features = train_x.select_dtypes(exclude = 'object').columns

categorical_features

numerical_features

########### all data #################
all_data = pd.concat([train_x,test_x],axis = 0)
all_data.shape

all_data.head(2)
all_data_cat = all_data.select_dtypes(include = 'object').columns
all_data_num = all_data.select_dtypes(exclude = 'object').columns


all_data_cat.shape, all_data_num.shape
all_data_cat
all_data_cat_df = pd.DataFrame(all_data[all_data_cat])
all_data_cat_df.shape
all_data_cat_df.isna().sum().sort_values(ascending = False)
all_data_cat_df.head()
all_data_num_df = pd.DataFrame(all_data[all_data_num])
all_data_num_df.head()
######### fine seperating the categorical values(nominal and ordinal) and numerical(continues and discrete) #################
all_data_num_df['MSSubClass'] = all_data_num_df['MSSubClass'].astype('str')
all_data_cat_df['MSSubClass'] = all_data_num_df['MSSubClass']
all_data_num_df.drop(['MSSubClass'], axis = 1, inplace = True)


all_data_num_df.isna().sum().sort_values(ascending = False)

all_data_num_df = pd.DataFrame(all_data[numerical_features])
all_data_num_df['LotFrontage'] = all_data_num_df['LotFrontage'].fillna(all_data_num_df['LotFrontage'].median())
all_data_num_df['LotFrontage'].isna().sum()

#################### CATEGORY CATEGORY ##############################
###### MSZoning fill ########
all_data_cat_df['MSZoning'].isna().sum()
all_data_cat_df['MSZoning'].fillna(method = 'ffill',inplace = True)

######## Filling Nan for categorical features ###########
features_cat = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType1','BsmtFinType2']
all_data_cat_df[features_cat] = all_data_cat_df[features_cat].fillna('None')

all_data_cat_df['MasVnrType'] = all_data_cat_df['MasVnrType'].fillna('None')

all_data_cat_df['Functional'] = all_data_cat_df['Functional'].fillna(method = 'ffill')

all_data_cat_df['Utilities'] = all_data_cat_df['Utilities'].fillna(method = 'ffill')

all_data_cat_df['KitchenQual'] = all_data_cat_df['KitchenQual'].fillna(method = 'ffill')

all_data_cat_df['SaleType'] = all_data_cat_df['SaleType'].fillna(method = 'ffill')

all_data_cat_df['Exterior2nd'] = all_data_cat_df['Exterior2nd'].fillna(method = 'ffill')

all_data_cat_df['Exterior1st'] = all_data_cat_df['Exterior1st'].fillna(method = 'ffill')

all_data_cat_df['Electrical'] = all_data_cat_df['Electrical'].fillna(method = 'ffill')
####### Street ########
all_data_cat_df['Street'].
#all_data_cat_df = all_data_cat_df.fillna('None')
all_data_cat_df.isna().sum().sort_values(ascending  = False)

label_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageCond','GarageQual']


label = LabelEncoder()
all_data_cat_df[label_features].shape
all_data_cat_df_label = label.fit_transform(all_data_cat_df[label_features])


############## NUMERICAL NUMERICAL #######################
all_data_num_df.isna().sum().sort_values(ascending = False)

all_data_num_df['GarageYrBlt'] = all_data_num_df['GarageYrBlt'].fillna(all_data_num_df['GarageYrBlt'].median())

all_data_num_df['MasVnrArea'] = all_data_num_df['MasVnrArea'].fillna(all_data_num_df['MasVnrArea'].median())

all_data_num_df['BsmtHalfBath'] = all_data_num_df['BsmtHalfBath'].fillna(method = 'ffill')

all_data_num_df['BsmtFullBath'] = all_data_num_df['BsmtFullBath'].fillna(method = 'ffill')

all_data_num_df['TotalBsmtSF'] = all_data_num_df['TotalBsmtSF'].fillna(method = 'ffill')

all_data_num_df['BsmtUnfSF'] = all_data_num_df['BsmtUnfSF'].fillna(method = 'ffill')

all_data_num_df['BsmtFinSF1'] = all_data_num_df['BsmtFinSF1'].fillna(method = 'ffill')

all_data_num_df['BsmtFinSF2'] = all_data_num_df['BsmtFinSF2'].fillna(method = 'ffill')

all_data_num_df['GarageCars'] = all_data_num_df['GarageCars'].fillna(method = 'ffill')

all_data_num_df['GarageArea'] = all_data_num_df['GarageArea'].fillna(method = 'ffill')

all_data_num_df['BsmtHalfBath'].unique()
numerical_features_na = ['BsmtHalfBath','BsmtFullBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','GarageCars','GarageArea']

all_data_num_df.isna().sum().sort_values(ascending = False)
all_data_cat_df_final = pd.get_dummies(all_data_cat_df)

all_data_cat_df_final.shape

all_data_final = pd.concat((all_data_cat_df_final,all_data_num_df), axis = 1)

all_data_final.shape


########## FINAL DATA ########### FINAL DATA ########### FINAL DATA ##############
train_final = all_data_final.iloc[:len(train_x)]
test_final = all_data_final.iloc[len(train_x):]
test_final.shape
train_final.shape
train_y.shape
################################################################################

####### model ###########
model = RandomForestRegressor()

model.fit(train_final,train_y)
model.score(train_final,train_y)

predict = model.predict(test_final)

############### submission ###########
submission = pd.read_csv('sample_submission.csv')
submission.tail()
submission = submission.drop(['SalePrice'],axis = 1)
submission['SalePrice'] = predict
submission.to_csv('submission5.csv', index = False)




############### GRADIENT BOOSTING REGRRESSOR #####################
gbr = GradientBoostingRegressor()
gbr.fit(train_final,train_y)
predict2 = gbr.predict(test_final)

submission2 = pd.read_csv('sample_submission.csv')
submission2.head()
submission2 = submission2.drop(['SalePrice'],axis = 1)
submission2['SalePrice'] = predict2
submission2.to_csv('submission6.csv',index = False)



####################### VOTING REGRESSOR ################################

r1 = GradientBoostingRegressor()
r3 = Lasso()
r2 = Ridge()
r5 = AdaBoostRegressor()
r4 = BaggingRegressor(n_estimators = 10,max_features = 215)
voting_regressor = VotingRegressor([('gbr',r1),('br',r4),('abr',r5),('l',r3),('r',r2)])
voting_regressor.fit(train_final,train_y)
predict3 = voting_regressor.predict(test_final)


### submissions ###
submission3.head()
submission3['SalePrice'] = predict3
submission3.head()
submission3.to_csv('submission9.csv',index = False)
pred_train = voting_regressor.predict(train_final)
error = mean_squared_error(train_y,pred_train)
final_error = np.sqrt(error)
final_error
 
