# ChatBot-via-GPT
## Project Dir
* **data** - data files for training revelence_model.py
* **ckpt_model** - relevence_model file
* **src** - source code
    - **app.py** - server file exposing chat API
    - **GPT.py** - A class calling openai API
    - **relevence_model.py** - model, dataset, inference
    - **trainer.py** - relevence model trainer
* **.gitignore**
* **README.md**

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
# response: return answers with relevent scores
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