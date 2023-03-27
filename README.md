# ChatBot-via-GPT
## Info
This project creates an application via openai GPT3.5 API to generate text based on user's input. It aims to return most relevant answers.

### Project Features:

* The application is based on python 
* Access to openai GPT3.5 api
* It will generate proper answers based on any user's input: topic, questions, prompt, etc..
* It handles some error cases.
* It handles spelling errors and grammatical errors in user's input.
* It will return sorted answers by relevance.
* Well documented


## Project Dir
* **data** - data files for training revelance_model.py
* **ckpt_model** - relevance_model file
* **src** - source code
    - **app.py** - server file exposing chat API
    - **GPT.py** - A class calling openai API
    - **relevance_model.py** - model, dataset, inference
    - **trainer.py** - relevance model trainer
* **.gitignore**
* **README.md**

### Download model file from following link and put it at **ckpt_model/latest.ckpt**

**model file url**: https://drive.google.com/drive/folders/1sl-QlkxevvUGLlB1hJHd0Gj6tz3Fw4ZK?usp=share_link
## Install and Run
```python
# create conda envs
conda create -n chatgpt python==3.10
# enter conda envs
conda activate chatgpt
# install python libraries
pip install -r requirements.txt
# run app file
python run src/app.py
```

## Call API

### 1. by python
```python
import requests
import json

url = "http://127.0.0.1:8000/query"

payload = json.dumps({
  "traceid": "001",
  "text": "QUESTION YOU WANT TO ASK"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

### 2. by curl
```bash
curl --location 'http://127.0.0.1:8000/query' \
--header 'Content-Type: application/json' \
--data '{
    "traceid": "001",
    "text": "QUESTION YOU WANT TO ASK"
}'
```

### 3. Example
```json
# request: put the question at text
{
    "traceid": "001",
    "text": "how are glacier caves formed?"
}
# response: return answers with relevant scores
{
    "content": "how are glacier caves formed?",
    "traceid": "001",
    "responses": [
        {
            "response": "Glacier caves are formed by meltwater flowing through or under a glacier. As the water melts away the ice, it creates passages and chambers that can eventually form into larger cave systems. The water is often fed by streams or rivers that flow onto or under the glacier, and as it moves through the ice, it erodes the surface and walls of the cave. Over time, these processes can create complex networks of tunnels and caverns within glaciers that are constantly changing due to shifting ice flows and melting patterns. Some glacier caves may also be formed by air currents that carve out hollow spaces in snowdrifts on top of a glacier before they become compacted into solid ice.",
            "score": 0.9995323419570923
        },
        ...,
        {
            "response": "Glacier caves are formed by the melting and refreezing of ice within a glacier. As water flows through the glacier, it erodes the ice and creates tunnels or caverns. These tunnels can grow larger over time as more water flows through them, creating complex cave systems. The movement of the glacier can also cause pressure on the walls of these caves, which can lead to collapse or reshaping of the cave structure. Additionally, warm air from outside can melt snow and ice near the entrance to a cave, leading to its enlargement. Glacier caves are often temporary structures because they depend on specific conditions for their formation and maintenance; when those conditions change (such as due to changes in temperature or precipitation patterns), glaciers may retreat or disappear altogether, taking their caves with them.",
            "score": 0.0006037128623574972
        },
        {
            "response": "Glacier caves are formed by a combination of glacial melting and erosion. As the glacier moves, it melts and water seeps into cracks and crevices in the ice. Over time, this water can create tunnels and caverns within the glacier.\n\nThe meltwater also helps to erode the walls of these tunnels, creating intricate patterns on the ice surfaces. Additionally, warm air from outside the glacier can enter through openings in its surface, causing further melting inside.\n\nGlacier caves can be unstable environments due to ongoing melting and shifting of the ice. It is important for visitors to exercise caution when exploring them.",
            "score": 0.000531199446413666
        }
    ],
    "status": true,
    "error": null
}
```

## Relevance model
**Input**: -> ['Question', 'Answer']

**output**: -> 0 or 1 label (if 'Question' and 'Answer' are revelent)

**Features**:
    
* bert context similarity (2 dims)
* question and answer sentence embeddings cosine simliarty (1 dim)
* question and answer sentence embeddings compress to 2-dim vector each and concat (4 dims)

7-dim features in total and feed in Fully connection layer and get 2-dim output, use the second-dim as its score 

## More Future Works
* Compress the relevance model to improve inference speed, or use onnx to accelerate model inference.
* Try more word-level feature on relevance model, such as, tfidf, bm25 etc..
* Train it on more data sources for training relevance model.
* Try more hyperparameters when training relevance model.
* Figure out all of the openai API error types, and handle exceptions more precisely.
* Multi-language relevance model.

## Data Source
```
@inproceedings{yang-etal-2015-wikiqa,
    title = "{W}iki{QA}: A Challenge Dataset for Open-Domain Question Answering",
    author = "Yang, Yi  and
      Yih, Wen-tau  and
      Meek, Christopher",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1237",
    doi = "10.18653/v1/D15-1237",
    pages = "2013--2018",
}
```