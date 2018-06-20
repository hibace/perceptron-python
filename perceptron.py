from __future__ import print_function
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
 
import scipy
import tensorflow as tf
import numpy as np
import csv
import random
 
scaler = StandardScaler()
 
def get_input_data(csv_path):
  result = []
 
  with open(csv_path, "r") as file_obj:
    reader = csv.reader(file_obj)
    next(reader, None)
    for row in reader:
      OPEN = float(row[4])
      CLOSE = float(row[7])
      result.append([OPEN, CLOSE])
 
  return result
 
def get_output_data(csv_path):
  result = []
 
  with open(csv_path, "r") as file_obj:
    reader = csv.reader(file_obj)
    next(reader, None)
    for row in reader:
      OPEN = float(row[4])
      CLOSE = float(row[7])
      COLOR = 1 if CLOSE - OPEN > 0 else 0
      result.append([COLOR, 1-COLOR])
 
  return result
 
 
input = get_input_data("sber_data.csv")
output = get_output_data("sber_data.csv")
 
# print(input);
# print(output);
 
X = input
y = np.array(output)
 
#  Параметры
learning_rate = 0.001 # определяет шаг значений для нахождения лучшего веса
training_epochs = 100 # количество прогона по данным
batch_size = 5
display_step = 1 # шаг прогона
 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Network Parameters
n_hidden_1 = 2 # количество признаков первого слоя
n_hidden_2 = 2 # количество признаков второго слоя
n_input = 2 # Данные катировок
n_classes = 2 # Категории
 
 
# Входные данные для графа
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
 
# Create model
def multilayer_perceptron(x, weights, biases):
    # 1 Скрытый слой с функцией активации
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 2 Скрытый слой с функцией активации
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    drop_out = tf.nn.dropout(layer_2, 0.1)
    out_layer = tf.matmul(drop_out, weights['out']) + biases['out']
    return out_layer
 
# Веса и смещения в переменных tf.Variable,
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
 
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
# Конструирование модели
pred = multilayer_perceptron(x, weights, biases)
 
# Определение потери
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 
# Инициализация
init = tf.global_variables_initializer()
 
# Запуск графа
with tf.Session() as sess:
    sess.run(init) #инициализация нормальным распределением
    # Тренировочный цикл
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train_scaled)/batch_size)
        X_batches = np.array_split(X_train_scaled, total_batch)
        Y_batches = np.array_split(y_train, total_batch)
        # Цикл по всем блокам
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Запустим оптимизацию  - op (backprop) и cost op (получение данных потери)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Подсчет точности
            avg_cost += c / total_batch
        # Отображение логов для каждого шага прогона
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Finished")
 
    # Тестирование модели
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Вычисление точности
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test_scaled, y: y_test}))
    global result
    result = tf.argmax(pred, 1).eval({x: X_test_scaled, y: y_test})
