import os
import openai
import language_tool_python
from relevance_model import QAInference

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
qaInference = QAInference('ckpt_model/latest.ckpt')

class GPT:
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.lang_tool = language_tool_python.LanguageTool('en-US')

    def chat(self, content):
        # correct spelling and grammer error
        content = self.lang_tool.correct(content)
        try:
            # call openai API
            completion = openai.ChatCompletion.create(
                presence_penalty=1,
                frequency_penalty=1,
                n=16,
                top_p=0.8,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            response = [i['message']['content'] for i in completion.choices]
            # relevence model sort the response
            score = qaInference.inference([[content]*len(response), response])
            ret = [{'response': r, 'score': float(s)} for r, s in zip(response, score)]
            ret.sort(key=lambda x: x['score'], reverse=True)
            return True, ret
        except Exception as e:
            return False, e


if __name__ == '__main__':
    gpt = GPT()
    res = gpt.chat('what is firendship')
    print(res)