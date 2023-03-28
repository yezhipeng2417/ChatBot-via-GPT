from sanic import Sanic
from sanic.response import json

from GPT import GPT
from sanic_cors import CORS  #加入扩展


gpt = GPT()
app = Sanic("GPTSever")
CORS(app)

@app.route('/query', methods=["POST"])
async def query(request):
    data = request.json
    print('request data', data)
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