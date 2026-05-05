import torch
import joblib
# 训练文件里：训练集 fit 完后保存
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
#固定种子

#========================数据处理=============================
df = pd.read_csv("D:/apytorchLearning/titanic/train.csv")
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
x = df.iloc[:, 1:]
y = df.iloc[:, 0]
#iloc是按行号取
#panda会自动把第一行作为列名，从第二行开始读
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#分割8：2训练测试，随机种子42方便复现

# 对 Age 列做 minmax 归一化
#df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
#df["Fare"] = (df["Fare"] - df["Fare"].min()) / (df["Fare"].max() - df["Fare"].min())
#max-min归一只能对连续数值特征使用

age_mean_by_title = x_train.groupby("Title")["Age"].mean()
train_age_mean = x_train["Age"].mean()
#只用训练集按照Title分组求Age均值，测试集只能使用训练集统计出来的均值
x_train["Age"] = x_train["Age"].fillna(x_train["Title"].map(age_mean_by_title)).fillna(train_age_mean)
x_train["Age"] = x_train["Age"].fillna(x_train["Title"].map(age_mean_by_title)).fillna(train_age_mean)
x_test["Age"] = x_test["Age"].fillna(x_test["Title"].map(age_mean_by_title)).fillna(train_age_mean)

joblib.dump(age_mean_by_title, "age_mean_by_title.pkl")
joblib.dump(train_age_mean, "train_age_mean.pkl")
#这两个变量用jiblib保存下来，预测的时候需要

#x_train = pd.get_dummies(x_train, columns=["Title"], drop_first=True)
#x_test = pd.get_dummies(x_test, columns=["Title"], drop_first=True)
x_train = x_train.drop("Title", axis = 1)
x_test =x_test.drop("Title", axis = 1)
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
#把Title做onehot，测试集的列顺序和列数量必须和训练集保持一致，因为可能称谓少一个就少一列
#所以如果少了的填0

scaler = MinMaxScaler()
#实例化方法
norm_cols = ["Age","Fare"]
x_train_scaled = x_train.copy()
#必须copy，不然是指向同一个dataframe，修改时会修改原数据
x_train_scaled[norm_cols] = scaler.fit_transform(x_train_scaled[norm_cols])
joblib.dump(scaler, "minmax_scaler.pkl")
x_test_scaled = x_test.copy()
x_test_scaled[norm_cols] = scaler.transform(x_test_scaled[norm_cols])
#sklearn里面的方法做归一化,fit保存了数据最大最小的尺度，测试集延用，不能fit

joblib.dump(x_train_scaled.columns.tolist(), "feature_columns.pkl")
#保存列顺序，使得预测的时候顺序一致，之前训练测试也做过



#=======================加载数据集===========================
class TitanicDataset(Dataset):
    def __init__(self,x_data,y_data):
    #不区分训练和测试，而是传参
        self.len = x_data.shape[0]
        self.x_data = torch.tensor(x_data.to_numpy(dtype="float32"))
        #values是去除列名变成纯numpy数组，和3范例的torch.from_numpy区别是，torch.from_numpy不能加dtype
        #后来我改成to_numpy,这样数据即使有 bool/int/float 混合，也会先统一转成真正的 float32 数组
        self.y_data = torch.tensor(y_data.to_numpy(dtype="float32").reshape(-1, 1))
        #reshape（-1自动算行数，1一列），之前的y_data直接转会变成一行（891，）但label格式是（891，1）视觉上是一竖列


    def __getitem__(self, index):
        #return self.x_data[index],self.y_data[index]
        #这里的data如果之前没有用torch.tensor转换成张量，还是panda daraframe，[]里就是填列名的地方，而不是行号
        
        return self.x_data[index],self.y_data[index]
        #这里之前加了iloc取行号，不过之前的格式已经在init改过了，就还按照原来的范例写

    def __len__(self):
        return self.len

train_dataset = TitanicDataset(x_train_scaled, y_train)
test_dataset =  TitanicDataset(x_test_scaled, y_test)


train_loader = DataLoader(  dataset = train_dataset,
                            batch_size=32,
                            shuffle = True,
                            num_workers=0)

test_loader = DataLoader(dataset = test_dataset,
                            batch_size=32,
                            shuffle = False,
                            num_workers=0)
#分别创建两个dataset和dataloader


#==========================模型=============================
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
    def forward(self,x):
        return self.net(x)

model = TitanicModel()

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)


#=========================训练============================
for epoch in range(500):

    epoch_loss = 0.0
    correct = 0
    total = 0
    for i,(inputs,labels) in enumerate(train_loader,0):
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        
        epoch_loss += loss.item()* inputs.size(0)
        #这里的loss是BCEWithLogitsLoss默认返回的是当前 batch 的平均 loss，所以*batchsize就是当前batch总损失

        prob = torch.sigmoid(y_pred)
        #把输出结果归一化，之前BCEWithLogitsLoss只是在算损失的内部归一化，y——pred并没有
        predicted = (prob >= 0.5).float()
        #(prob >= 0.5)输出的是0 1，转化为浮点数是1.0，0.0
        correct += (predicted == labels).sum().item()
        #把预测正确为1的加起来，正确数量等于加起来的数量，sum输出的是张量所以要item
        total += labels.size(0)
        #计算出总测试样本数

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(epoch,"avg loss:" ,epoch_loss/total)
        print("accuracy:", correct / total)


#==========================测试=============================
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
#不更新权重
    for i,(inputs,labels) in enumerate(test_loader,0):
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        test_loss += loss.item()* inputs.size(0)

        prob = torch.sigmoid(y_pred)
        predicted = (prob >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)


print("avg test loss:",test_loss / total)
print("test accuracy:", correct / total)

torch.save(model.state_dict(), "titanic_model.pth")