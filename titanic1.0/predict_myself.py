import torch
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
#固定种子

#========================数据处理=============================
df = pd.read_csv("D:/apytorchLearning/titanic/test.csv")
#如果没有列名这样写：df = pd.read_csv("data.csv", header=None)

df =df.drop("PassengerId", axis = 1)
df =df.drop("Ticket", axis = 1)
df =df.drop("Cabin", axis = 1)
#axis=1按列删除

df["Title"] = df["Name"].str.extract(r"(Mrs|Mr|Miss|Master)").fillna("Other")
#新建一列Title,extract挖取字符串中需要的部分，fillna把空的部分填other
#Age 的缺失值要在划分训练/测试之后再填，避免测试集信息参与训练
#Title 的 onehot 也放到划分之后处理，保证测试集按训练集列对齐

df =df.drop("Name", axis = 1)

df["Sex"] = df["Sex"].map({"male":0,"female":1})
#map映射，把内容更换
df["Embarked"] = df["Embarked"].fillna('S')
df = pd.get_dummies(df, columns=["Embarked","Pclass"], drop_first=True)
#对港口列做one_hot编码，删除第一列避免冗余
#注意检查列数变化，模型第一层维度改变
#dtype必须转换成torch.float32可以处理的int，不加的话是object 

#panda必须一列都是同一种格式，字符串或者数字，"0"是字符串要注意

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1)

df.to_csv("clean_titanic.csv",index = False)
#index不生成序列，如果=pandas自动给每行加行号
#保存在运行位置（打开的folder）


#=======================划分测试集训练集=====================

age_mean_by_title = joblib.load("age_mean_by_title.pkl")
train_age_mean = joblib.load("train_age_mean.pkl")
#之前保存的训练变量
df["Age"] = df["Age"].fillna(df["Title"].map(age_mean_by_title)).fillna(train_age_mean)


#x_train = pd.get_dummies(x_train, columns=["Title"], drop_first=True)
#x_test = pd.get_dummies(x_test, columns=["Title"], drop_first=True)
df = df.drop("Title", axis = 1)

scaler = MinMaxScaler()
#实例化方法
norm_cols = ["Age","Fare"]
df_scaled = df.copy()
#必须copy，不然是指向同一个dataframe，修改时会修改原数据
scaler = joblib.load("minmax_scaler.pkl")
df_scaled[norm_cols] = scaler.transform(df_scaled[norm_cols])
#sklearn里面的方法做归一化,fit保存了数据最大最小的尺度，测试集延用，不能fit

feature_columns = joblib.load("feature_columns.pkl")
df_scaled = df_scaled.reindex(columns=feature_columns, fill_value=0)
#保证预测的列顺序和训练相同，不然代入就会错

#=======================加载数据集===========================
class TitanicDataset(Dataset):
    def __init__(self,df_scaled):
    #不区分训练和测试，而是传参
        self.len = df_scaled.shape[0]
        self.x_data = torch.tensor(df_scaled.to_numpy(dtype="float32"))
        #values是去除列名变成纯numpy数组，和3范例的torch.from_numpy区别是，torch.from_numpy不能加dtype
        #后来我改成to_numpy,这样数据即使有 bool/int/float 混合，也会先统一转成真正的 float32 数组
        


    def __getitem__(self, index):
        #return self.x_data[index],self.y_data[index]
        #这里的data如果之前没有用torch.tensor转换成张量，还是panda daraframe，[]里就是填列名的地方，而不是行号
        
        return self.x_data[index]
        #这里之前加了iloc取行号，不过之前的格式已经在init改过了，就还按照原来的范例写

    def __len__(self):
        return self.len

predict_dataset = TitanicDataset(df_scaled)



predict_loader = DataLoader(dataset = predict_dataset,
                            batch_size=32,
                            shuffle = False,
                            num_workers=0)





#=========================预测============================
class TitanicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(11, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)

model = TitanicModel()


model.load_state_dict(torch.load("titanic_model.pth"))
model.eval()

predictions = []
with torch.no_grad():
#不更新权重
    for i,inputs in enumerate(predict_loader,0):
        y_pred = model(inputs)
        prob = torch.sigmoid(y_pred)
        predicted = (prob >= 0.5).int()
        predictions.extend(predicted.view(-1).tolist())


passenger_ids = pd.read_csv("test.csv")["PassengerId"]
submission = pd.DataFrame({"PassengerId": passenger_ids,"Survived": predictions})
submission.to_csv("submission.csv", index=False)