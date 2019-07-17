import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence, text
from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import KerasModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

max_features = 1000

@artifacts([
    KerasModelArtifact('model'),
    PickleArtifact('word_index')
])
@env(conda_dependencies=['tensorflow', 'numpy', 'pandas'])
class TextClassificationService(BentoService):
   
    def word_to_index(self, word):
        if word in self.artifacts.word_index and self.artifacts.word_index[word] <= max_features:
            return self.artifacts.word_index[word]
        else:
            return self.artifacts.word_index["<UNK>"]
    
    def preprocessing(self, text_str):
        sequence = text.text_to_word_sequence(text_str)
        return list(map(self.word_to_index, sequence))
    
    @api(JsonHandler)
    def predict(self, parsed_json):
        if type(parsed_json) == list:
            input_data = list(map(self.preprocessing, parsed_json))
        else: # expecting type(parsed_json) == dict:
            input_data = [self.preprocessing(parsed_json['text'])]

        input_data = sequence.pad_sequences(input_data,
                                            value=self.artifacts.word_index["<PAD>"],
                                            padding='post',
                                            maxlen=80)

        return self.artifacts.model.predict_classes(input_data)
