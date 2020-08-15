from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
from datetime import datetime

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
		n_features = 3
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=np.int)

		for i, sample in enumerate(data_file):
			data[i] = np.asarray(sample[:-1], dtype=np.float64)
			target[i] = np.asarray(sample[-1], dtype=np.int)
	
	return Bunch(data=data, target=target)


def on_message(client, userdata, message):

	"""Recibe las tomas de 25 en 25"""
	f = open('csv/csv_temp.csv','w')
	f.write(str(message.payload.decode("utf-8")))
	f.close()


	"""Datos que se toman como base"""
	dataset = load_dataset('csv/csv_ismael.csv')

	X = dataset.data
	
	y = dataset.target

	
	algoritmo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
	
	"""KFold"""
	kf = KFold(n_splits=5)
	
	for train_index, test_index in kf.split(X):

	    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
	    algoritmo.fit(X_train, y_train)

	"""Se cargan los datos con las features"""
	dataset2 = load_dataset('csv/csv_temp.csv')

	X_temp = dataset2.data

	y_temp = dataset2.target

	y_pred = algoritmo.predict(X_temp)

	"""Se muestra la predicción para cada muestra y se devuelve la mayoritaria"""
	print(y_pred)
	c = Counter(y_pred)
	
	value, count = c.most_common()[0]
	if value == 0:
		print("RELAJADO")
		client.publish("stress_results", "0")
	if value == 1:
		print("ESTRESADO")
		client.publish("stress_results", "1")
	




client.on_message=on_message

client.connect("broker.hivemq.com")

client.subscribe("stress_data")

client.loop_forever()