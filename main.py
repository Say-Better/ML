import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel
import recommender.recommend_word_card as recommend_word_card
import functions_framework

@functions_framework.http
def start(request):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    # 'name' 키에 해당하는 JSON 데이터에 접근
    request_data = request.json
    name = request_data.get('name', '')  # 'name' 키가 없을 경우 빈 문자열 반환
    
    results = recommend_word_card.predict_word_card(name)
    return (results, 200)
