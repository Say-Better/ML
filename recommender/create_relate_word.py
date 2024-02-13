import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel

def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "",
    tuned_model_name: str = "",
    ) :

    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = ChatModel.from_pretrained(model_name)

    parameters = {
            "temperature": 0.2,  
            "top_p": 0.95,  
            "top_k": 40,  
        }

    chat =  model.start_chat(
        context="내가 어떤 상황을 말해주면 너는 그 상황에 어울리는 단어 10가지를 list형태로 답해줘야해. 문장의 끝에 오는 단어라면 '요'의 어말어미를 써줘",
        examples=[
            InputOutputTextPair(
                input_text="화장실이 가고 싶은 상황",
                output_text="['화장실', '변기', '소변', '대변', '가고싶어요', '어디에요', '어디있어요', '알려주세요', '위치', '세면대']",
            ),
        ],
    )
    response = chat.send_message(
        content, **parameters
    )
    print(f"Response from Model: {response.text}")

    response = (response.text).replace("'", '')[2:-1].split(',')

    return response