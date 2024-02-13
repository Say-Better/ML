import recommender.create_relate_word as create_relate_word
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd



def predict_word_card(user_input:str):

  # 사용가능한 단어데이터 경로
  file_path1 = "./unique_word_cards.csv"
  word_df = pd.read_csv(file_path, header=None, names=['word'])

  # 한국어 단어 카드
  word_cards = word_df['word'].tolist()

  total_results = []

  # KoBERT 모델 로드
  model_id = 'kykim/bert-kor-base'
  tokenizer = BertTokenizer.from_pretrained(model_id)
  model = TFBertModel.from_pretrained(model_id)

  # 단어 카드의 임베딩 불러오기
  with open('./word_card_embed.npy', 'rb') as f:
    word_card_embeddings = np.load(f)

  # word_card_embeddings = np.array([model(tokenizer(card, return_tensors='tf'))['last_hidden_state'][:, 0, :] for card in word_cards])

  model_response = create_relate_word.predict_large_language_model_sample('wise-imagery-410607', "chat-bison@002", 0.2, 256, 0.8, 40, user_input, "us-central1")
  print(model_response)
  total_res = []
  for res_card in model_response:

    # 키워드의 임베딩 생성
    user_keyword_embedding = model(tokenizer(res_card, return_tensors='tf'))['last_hidden_state'][:, 0, :]
    cosine_similarities = []

    # 코사인 유사도 계산
    cosine_similarities = tf.keras.losses.cosine_similarity(user_keyword_embedding, word_card_embeddings).numpy()

    combined_data = list(zip(word_cards, cosine_similarities.tolist()))
    # 유사도가 높은 상위 10개 단어 카드 선택
    top_k = 3
    top_results = sorted(combined_data, key=lambda x: x[1], reverse=False)[:top_k]

    # 결과 출력
    # for idx in top_results:
    #     print(f"{idx[0]} (유사도: {idx[1]:.4f})")

  res_str = ''
  for card in total_res:
    for card_name in card:
      res_str += card_name[0] + ' '

  return res_str
