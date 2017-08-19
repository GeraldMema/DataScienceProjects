#libraries import
import pandas as pd
import matplotlib.pyplot as plt

#datasets
dataTrainPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\DataMining\\ted2\\train.tsv'

#read train dataset
df = pd.read_csv(dataTrainPath, sep='\t')

#plot
#categorial
df.reset_index().pivot('index','Attribute1','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute3','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute4','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute6','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute7','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute8','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute9','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute10','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute11','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute12','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute14','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute15','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute16','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute17','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute18','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute19','Label').hist(alpha=0.7)
df.reset_index().pivot('index','Attribute20','Label').hist(alpha=0.7)
#numerical values
df.boxplot(column='Attribute2',by='Label')
df.boxplot(column='Attribute5',by='Label')
df.boxplot(column='Attribute13',by='Label')


plt.show()