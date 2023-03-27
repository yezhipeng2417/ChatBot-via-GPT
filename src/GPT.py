import os
import openai
import language_tool_python
from relevence_model import QAInference

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
qaInference = QAInference('ckpt_model/latest.ckpt')

class GPT:
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def chat(self, content):
        # correct spelling and grammer error
        my_tool = language_tool_python.LanguageTool('en-US')
        content = my_tool.correct(content)
        try:
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
            return True, response
        except Exception as e:
            return False, e


if __name__ == '__main__':
    gpt = GPT()
    res = gpt.chat('what is firendship')
    print(res)