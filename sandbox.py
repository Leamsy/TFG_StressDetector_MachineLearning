from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve,GridSearchCV
from io import StringIO
import paho.mqtt.client as mqtt
import numpy as np
import csv
from sklearn.utils import Bunch
from joblib import dump, load
import pandas as pd
import os
import glob
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

client = mqtt.Client("StressReader")

print("Ready")


def file_len(fname):

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def load_dataset(name):

	with open(name) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = (file_len(name)-1)
		n_features = 24
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=np.int)

		for i, sample in enumerate(data_file):
			data[i] = np.asarray(sample[:-1], dtype=np.float64)
			target[i] = np.asarray(sample[-1], dtype=np.int)
	
	return Bunch(data=data, target=target)


def on_message(client, userdata, message):

	f = open('csv/csvfile_temp.csv','w')
	f.write(str(message.payload.decode("utf-8")))
	f.close()


	dataset = load_dataset('csv/csvfile.csv')

	X = dataset.data

	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	algoritmo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

	algoritmo.fit(X_train, y_train)


	dataset2 = load_dataset('csv/csvfile_temp.csv')

	X_temp = dataset2.data

	y_temp = dataset2.target
	

	y_pred = algoritmo.predict(X_temp)

	print(y_temp)
	print(y_pred)

	c = Counter(y_pred)

	value, count = c.most_common()[0]
	if value == 0:
		print("RELAJADO")
		client.publish("stress_results", "0")
	if value == 1:
		print("ESTRESADO")
		client.publish("stress_results", "1")

	
	precision = precision_score(y_temp, y_pred)
	print('Precision del modelo:', precision)

	matriz = confusion_matrix(y_temp, y_pred)
	print('Matriz de Confusion: ')
	print(matriz)
	

def feature_selection():

	df = pd.read_csv('csv/matriz.csv', index_col=24)
	X = df
	y = df.index
	print(X)
	print(y)

	reg = LassoCV()
	reg.fit(X, y)
	print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
	print("Best score using built-in LassoCV: %f" %reg.score(X,y))
	coef = pd.Series(reg.coef_, index = X.columns)

	print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
	print(coef)
	imp_coef = coef.sort_values()
	import matplotlib
	matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
	imp_coef.plot(kind = "barh")
	plt.title("Lasso")
	plt.savefig('foo.png')


def grid():

	dataset = load_dataset('matriz_def.csv')

	X = dataset.data

	y = dataset.target

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()
	X_scaled = scaler.fit_transform(X)

	k_range = list(range(1,5))
	weight_options = ["uniform", "distance"]
	param_grid = dict(n_neighbors = k_range, weights = weight_options)

	knn = KNeighborsClassifier()

	grid = GridSearchCV(knn, param_grid, cv = 5, scoring = 'accuracy')
	grid.fit(X_scaled,y)

	print (grid.best_score_)


def sliding_windows_with_50_perc_overlapping(d, w=0.5):
	t = int(w / 2)
	r = np.arange(len(d))
	s = r[::t]
	z = list(zip(s, s + w))
	fz = list()
	ld = len(d)
	for z1, z2 in z:
		if z1 > ld or z2 > ld:
			break
		fz.append((z1, z2))
	return fz


def sliding_windows_with_50_perc_overlapping_splitting_events(d, w=0.5):
	fz = list()
	df_list = []
	df_final = None

	fz = sliding_windows_with_50_perc_overlapping(d, w)
	for w_i in fz:
		df_list.append(d[w_i[0]:w_i[1]])
	return df_list

def exp_plot():
	df = pd.read_csv("experimento/experimento_2.csv", index_col=3)

	cont = []
	cont_2 = ["Inicio", "1º parte (relajante)", "2º parte (estresante)", "3º parte (relajante)"]
	respuesta = [1,1,8,3]

	for i in range (len(df)):
		cont.append(i)

	fig = px.line(df, x = cont, y = 'pulsaciones', title='Sujeto 2: Pulsaciones')
	fig.show()

	fig = px.line(df, x = cont, y = ' gsr', title='Sujeto 2: GSR')
	fig.show()

	fig = px.line(df, x = cont, y = ' temperatura', title='Sujeto 2: Temperatura')
	fig.show()

	fig = px.line(df, x = cont_2, y = respuesta, title='Sujeto 2: Respuesta en formulario')
	fig.show()





df = pd.read_csv('experimento/csvfile_antonio.csv', index_col=3)
df0 = df.loc[[0]]
df1 = df.loc[[1]]
w = 0.2
hz = 5
w_samples_n = int(hz * w)

f = open('matriz_def.csv','a')

f2 = open('wind_temp.csv','w')
f2.write("pulsaciones, gsr, temperatura\n")
f2.write(str(df0.iloc[0,0]))
f2.write(",")
f2.write(str(df0.iloc[0,1]))
f2.write(",")
f2.write(str(df0.iloc[0,2]))
f2.write(",0")

for i in range(1, len(df0)):
	print(i)
	if i%100 == 0 or i==len(df0)-1:
		f2.close()
		df_temp = pd.read_csv('wind_temp.csv', index_col=3)

		res = sliding_windows_with_50_perc_overlapping_splitting_events(df_temp, 10)

		f3 = open('mean_temp.csv','w')
		f3.write("pulsaciones_media, gsr_media, temperatura_media, pulsaciones_mediana, gsr_mediana, temperatura_mediana, pulsaciones_std, gsr_std, temperatura_std, "
					"pulsaciones_min, gsr_min, temperatura_min, pulsaciones_max, gsr_max, temperatura_max, pulsaciones_minmax, gsr_minmax, temperatura_minmax, pulsaciones_skew, gsr_skew, temperatura_skew, pulsaciones_kurt, gsr_kurt, temperatura_kurt")


		for i in range(len(res)):
			
			dataset = res[i]
			
			for i in range(len(dataset)):
				
				f3.write("\n" + str(dataset.mean()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.mean()[j]))

				f3.write("," + str(dataset.median()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.median()[j]))

				f3.write("," + str(dataset.std()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.std()[j]))

				f3.write("," + str(dataset.min()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.min()[j]))

				f3.write("," + str(dataset.max()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.max()[j]))

				minmax = dataset.max()[0] - dataset.min()[0]
				f3.write("," + str(minmax))
				for j in range(1, 3):
					minmax = dataset.max()[j] - dataset.min()[j]
					f3.write("," + str(minmax))

				f3.write("," + str(dataset.skew()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.skew()[j]))

				f3.write("," + str(dataset.kurt()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.kurt()[j]))

				f3.write(",0")

		f3.close()
		df_mean = pd.read_csv('mean_temp.csv', index_col=24)

		f.write("\n" + str(df_mean.mean()[0]))
		for j in range(1, 24):
			f.write("," + str(df_mean.mean()[j]))
		f.write(",0")
		
		f2 = open('wind_temp.csv','w')
		f2.write("pulsaciones, gsr, temperatura")


	f2.write("\n")
	f2.write(str(df0.iloc[i,0]))
	f2.write(",")
	f2.write(str(df0.iloc[i,1]))
	f2.write(",")
	f2.write(str(df0.iloc[i,2]))
	f2.write(",0")




f2 = open('wind_temp.csv','w')
f2.write("pulsaciones, gsr, temperatura\n")
f2.write(str(df1.iloc[0,0]))
f2.write(",")
f2.write(str(df1.iloc[0,1]))
f2.write(",")
f2.write(str(df1.iloc[0,1]))
f2.write(",1")

for i in range(1, len(df1)):
	print(i)
	if i%100 == 0 or i==len(df1)-1:
		f2.close()
		df_temp = pd.read_csv('wind_temp.csv', index_col=3)

		res = sliding_windows_with_50_perc_overlapping_splitting_events(df_temp, 10)

		f3 = open('mean_temp.csv','w')
		f3.write("pulsaciones_media, gsr_media, temperatura_media, pulsaciones_mediana, gsr_mediana, temperatura_mediana, pulsaciones_std, gsr_std, temperatura_std, "
					"pulsaciones_min, gsr_min, temperatura_min, pulsaciones_max, gsr_max, temperatura_max, pulsaciones_minmax, gsr_minmax, temperatura_minmax, pulsaciones_skew, gsr_skew, temperatura_skew, pulsaciones_kurt, gsr_kurt, temperatura_kurt")


		for i in range(len(res)):
			
			dataset = res[i]
			
			for i in range(len(dataset)):
				
				f3.write("\n" + str(dataset.mean()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.mean()[j]))

				f3.write("," + str(dataset.median()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.median()[j]))

				f3.write("," + str(dataset.std()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.std()[j]))

				f3.write("," + str(dataset.min()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.min()[j]))

				f3.write("," + str(dataset.max()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.max()[j]))

				minmax = dataset.max()[0] - dataset.min()[0]
				f3.write("," + str(minmax))
				for j in range(1, 3):
					minmax = dataset.max()[j] - dataset.min()[j]
					f3.write("," + str(minmax))

				f3.write("," + str(dataset.skew()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.skew()[j]))

				f3.write("," + str(dataset.kurt()[0]))
				for j in range(1, 3):
					f3.write("," + str(dataset.kurt()[j]))

				f3.write(",1")
		
		f3.close()
		df_mean = pd.read_csv('mean_temp.csv', index_col=24)

		f.write("\n" + str(df_mean.mean()[0]))
		for j in range(1, 24):
			f.write("," + str(df_mean.mean()[j]))
		f.write(",1")
		
		f2 = open('wind_temp.csv','w')
		f2.write("pulsaciones, gsr, temperatura")


	f2.write("\n")
	f2.write(str(df1.iloc[i,0]))
	f2.write(",")
	f2.write(str(df1.iloc[i,1]))
	f2.write(",")
	f2.write(str(df1.iloc[i,2]))
	f2.write(",1")

f.close()