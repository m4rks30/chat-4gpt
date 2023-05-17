# chat-4gpt
#A bot similar to GPT chat with graphical environment
pip install tensorflow

## python
```python 
# Load the dataset
dataset, info = tfd.load('conv_ai/2', with_info=True)
train_data = dataset['train']
validation_data = dataset['validation']

``` 
``` 

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tf.keras.layers.experimental.preprocessing.TextVectorization
    (max_tokens=8000),
    tf.keras.layers.Embedding(8000, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

```

```

model.fit(train_data,

          validation_data=validation_data,
          epochs=10)


```
