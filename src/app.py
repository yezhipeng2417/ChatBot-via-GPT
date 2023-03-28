from sanic import Sanic
from sanic.response import json

from GPT import GPT

gpt = GPT()

app = Sanic("GPTSever")

@app.route('/query', methods=["POST"])
async def query(request):
    data = request.json
    # user input
    text = data['text']
    # input id from upstream
    traceid = data['traceid']
    # call openai api
    status, responses = gpt.chat(text)
    if status:
        ret = {
            'content': text,
            'traceid': traceid,
            'responses': responses,
            'status': status,
            'error':None
        }
    else:
        ret = {
            'content': text,
            'traceid': traceid,
            'responses': None,
            'status': status,
            'error':str(responses)
        }
    return json(ret)

if __name__ == '__main__':
    app.run(debug=True)  