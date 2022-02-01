import numpy as np
import scipy.special as spc

class NeuralNetwork:
    '''Класс нейронной сети'''

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        '''Инициализация нейронной сети'''

        #Количество узлов входного, скрытого и выходного слоёв
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        #Коэффициент обучения
        self.lr = learning_rate

        #Веса между входным и скрытым и скрытым и выходным слоями
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        #Функция активации
        self.activation_function = lambda x: spc.expit(x)


    def train(self, inputs_list, target_list):
        '''Тренировка нейронной сети'''

        #Преобразование списка входных и целевых значений в двумерные массивы
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        #Расчёт входных и выходный сигналов скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #Расчёт входных и выходных сигналов последнего слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #Ошибка выходного, и скрытого слоёв слоя
        output_errors = (targets - final_outputs)
        hidden_errors = np.dot(self.who.T, output_errors)

        #К-орректировка весов связей между выходным и скрытым слоями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)),
                                     np.transpose(hidden_outputs))

        #Корректировка весов связей между входным и скрытым слоем
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)),
                               np.transpose(inputs))


    def query(self, inputs_list):
        '''Опрос нейронной сети'''

        inputs = np.array(inputs_list, ndmin=2).T

        #Расчёт входных и выходный сигналов скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #Расчёт входных и выходных сигналов выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

#Количество входных, скрытых и выходных узлов
inodes = 784
hnodes = 200
onodes = 10

#Коэффициент обучения
lr = 0.2

n = NeuralNetwork(inodes, hnodes, onodes, lr)

#Получение тренировочных данных
training_data_file = open('mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epoch = 5

for e in range(epoch):
    for record in training_data_list:

        #Подготовка входных данных
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

        #Подготовка выходных данных
        targets = np.zeros(onodes) + 0.01
        targets[int(all_values[0])] = 0.99

        #Тренировка сети
        n.train(inputs, targets)

#Подготовка тестовых данных
test_data_file = open('mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    #Преобразование строки в список чисел и получение корректного значения
    all_values = record.split(',')
    correct_label = int(all_values[0])

    #Передача входных данных нейросети и получение выходных её данных
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)

    #Получение ответа сети из её выходного массива
    label = np.argmax(outputs)

    print(correct_label, '- истинное значение;', label, '- ответ сети')

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)


scorecard_array = np.asarray(scorecard)
print ("Эффективность -", scorecard_array.sum() / scorecard_array.size * 100, " %")