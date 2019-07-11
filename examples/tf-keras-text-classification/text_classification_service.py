import pandas as pd
import numpy as np
from tensorflow import keras
from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TfKerasModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

@artifacts([
    TfKerasModelArtifact('model'),
    PickleArtifact('word_index')
])
@env(conda_dependencies=['tensorflow', 'numpy', 'pandas'])
class TextClassificationService(BentoService):
   
    def word_to_index(self, word):
        if word in self.artifacts.word_index:
            return self.artifacts.word_index[word]
        else:
            return self.artifacts.word_index["<UNK>"]
    
    def preprocessing(self, text):
        sequence = keras.preprocessing.text.text_to_word_sequence(text)
        return list(map(self.word_to_index, sequence))
    
    @api(JsonHandler)
    def predict(self, parsed_json):
        if type(parsed_json) == list:
            input_data = list(map(self.preprocessing, parsed_json))
        
        else: # expecting type(parsed_json) == dict:
            input_data = [self.preprocessing(parsed_json['text'])]

#             input_data = self.preprocessing(parsed_json['text'])
#             input_data = np.expand_dims(input_data, 0)

        input_data = keras.preprocessing.sequence.pad_sequences(input_data,
                                                                value=self.artifacts.word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=80)
#         input_data = keras.preprocessing.sequence.pad_sequences(input_data,
#             value=self.artifacts.word_index["<PAD>"],
#             padding='post',
#             maxlen=256)

        return self.artifacts.model.predict_classes(input_data)
