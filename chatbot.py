#Importa las bibliotecas necesarias:
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os



#Carga del archivo JSON:
#En el archivo 'chatbot.py', debes cargar el archivo JSON y preparar los datos de entrenamiento. 
# Puedes hacer esto con el siguiente código:

with open('C:\\Users\\sanne\\OneDrive\\Escritorio\\Programming\\python\\chatbot\\datos_entrenamiento.json') as archivo:
    datos_entrenamiento = json.load(archivo)

    print(os.path.abspath('datos_entrenamiento.json'))
    



# Extrae las preguntas y respuestas de los datos de entrenamiento y 
# conviértelas en listas separadas utilizando las siguientes líneas de código:
preguntas = []
respuestas = []

for pregunta in datos_entrenamiento['preguntas']:
    preguntas.append(pregunta['texto'])
    print(pregunta['texto'])

for respuesta in datos_entrenamiento['respuestas']:
    respuestas.append(respuesta['texto'])
    print(respuesta['texto'])

    

#Define una función para preprocesar los datos y convertirlos 
# en un formato que el modelo de TensorFlow pueda entender. 
# Esta función se encargará de tokenizar las preguntas y respuestas, 
# asignar un número a cada palabra y rellenar las secuencias para que tengan la misma longitud. 
# Aquí te muestro un ejemplo de cómo podría verse esta función:

def preprocesar_datos(preguntas, respuestas):
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(preguntas + respuestas)
    preguntas_seq = tokenizer.texts_to_sequences(preguntas)
    respuestas_seq = tokenizer.texts_to_sequences(respuestas)
    preguntas_seq = keras.preprocessing.sequence.pad_sequences(preguntas_seq, padding='post')
    respuestas_seq = keras.preprocessing.sequence.pad_sequences(respuestas_seq, padding='post')
    return preguntas_seq, respuestas_seq, tokenizer


#Utiliza la función "preprocesar_datos" para preprocesar los datos de entrenamiento 
# utilizando las siguientes líneas de código:


preguntas_seq, respuestas_seq, tokenizer = preprocesar_datos(preguntas, respuestas)
print(preguntas_seq)
print(respuestas_seq)


#Define el modelo de TensorFlow utilizando la siguiente línea de código:

max_input_len = 100  # Definición de la variable

modelo = keras.Sequential([
    keras.layers.Embedding(len(tokenizer.word_index) + 1, 128, input_length = max_input_len ),
    keras.layers.LSTM(128),
    keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])
modelo.summary()




#Este modelo utiliza una capa de Embedding para convertir las palabras en 
# vectores, una capa LSTM para procesar las secuencias y una capa Dense para generar las respuestas.

#Compila el modelo utilizando la siguiente línea de código:
modelo.compile(optimizer='adam', loss='categorical_crossentropy')
print(modelo.optimizer)
print(modelo.loss)

#Entrena el modelo utilizando los datos de entrenamiento utilizando las siguientes líneas de código:

modelo.fit(preguntas_seq, keras.utils.to_categorical(respuestas_seq, num_classes=len(tokenizer.word_index) + 1), epochs=500)
print(preguntas_seq.shape)
print(keras.utils.to_categorical(respuestas_seq, num_classes=len(tokenizer.word_index) + 1).shape)
#Define una función para generar respuestas a partir de las preguntas del usuario 
# utilizando el modelo entrenado y el tokenizer. Esta función tomará una pregunta como entrada, 
# la preprocesará utilizando el tokenizer y la alimentará al modelo para generar una respuesta. 
# Aquí te muestro un ejemplo de cómo podría verse esta función:
def generar_respuesta(pregunta):
    pregunta_seq = tokenizer.texts_to_sequences([pregunta])
    pregunta_seq = keras.preprocessing.sequence.pad_sequences(pregunta_seq, padding='post', maxlen=preguntas_seq.shape[1])
    respuesta_seq = modelo.predict(pregunta_seq)
    respuesta_seq = np.argmax(respuesta_seq, axis=-1)[0]
    respuesta = ''
    for palabra, index in tokenizer.word_index.items():
        if index == respuesta_seq:
            respuesta = palabra
            break
    return respuesta

#Finalmente, agrega un bucle que permita al usuario hacer preguntas y genere respuestas 
# utilizando la función "generar_respuesta". 
# Aquí te muestro un ejemplo de cómo podría verse este bucle:
while True:
    pregunta = input('Pregunta: ')
    respuesta = generar_respuesta(pregunta)
    print('Respuesta: ', respuesta)










