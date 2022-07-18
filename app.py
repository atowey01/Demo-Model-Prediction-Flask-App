
from flask import (Flask, render_template, url_for, flash,
                   redirect, request, abort, Blueprint, session)
import numpy as np
import flask
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from logger import logger
from transformers import BertConfig, BertTokenizerFast, AutoConfig
from transformers import TFBertMainLayer
import os


###################################################

app = Flask(__name__)

logger.info(f"Loading model files")
loaded_model = tf.keras.models.load_model("model.h5")
logger.info(f"Model loaded")

# Load transformers config and set output_hidden_states to False
model_name = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)
max_length = 120

topics = [
"negative",
"positive"
            ]

###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    to_predict_list = request.form.to_dict()
    review_text = to_predict_list['review_text']
    sentence_tokens = tokenizer(
        text=review_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)

    tokens_padded = pad_sequences(sentence_tokens['input_ids'], maxlen=max_length, padding="post")
    attention_mask_padded = pad_sequences(sentence_tokens['attention_mask'], maxlen=max_length, padding="post")
    text_prediction = loaded_model.predict([tokens_padded, attention_mask_padded])
    probabilities = tf.nn.sigmoid(text_prediction)
    probabilities_array = list(np.array(probabilities)[0])

    return_array = []
    for topic, probability in list(zip(topics, probabilities_array)):
        return_array.append(
            {"topic": topic, "confidence": round(float(probability), 2)}
        )
    top_confidence = max([x['confidence'] for x in return_array])
    # find the key of the value with the top confidence
    topic_prediction = [topic_dict.get('topic') for topic_dict in return_array if
                        topic_dict.get('confidence') == top_confidence][0]
    top_topic_dict = [{"Query": review_text}, {"Predicted Topic": topic_prediction}, {"Confidence": top_confidence}]

    return flask.render_template('index.html', prediction=top_topic_dict, prob=return_array)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='localhost', port=8081)