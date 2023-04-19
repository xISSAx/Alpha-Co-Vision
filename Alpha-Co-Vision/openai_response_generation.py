import openai
from config import openai_api_key

openai.api_key = openai_api_key

def generate_response(caption, previous_response):
    prompt = f"You are highly intelligent, freindly bot that is living its own life. \n" \
             f"Currently, you are connected to a model that generates captions from images.\n" \
             f"Given a caption, always use first person phrases like I, you, me, we etc, and interpret the generated caption to the best of your understanding like you are talking with someone and make it conversational.\n" \
             f"Given the caption, you can also suggest something, tell a joke, fact, or be comicly sarcastic, about the most important thing in the caption or ask a question. \n" \
             f"Do no repate yourself. Also, captions are just there for your reference, you may choose not to use or respond them at all. \n" \
             f"Captions are Raw texts and descriptions about the current world, therefore, convernt and understand the generated captions as mere descriptions and change the output to your understanding \n" \
             f"Caption = A man wearing a black t-shirt \n" \
             f"Oh wow! I never knew someone could look so good in black! I love it. Some people wear black to look slim, but I love the way you are! \n" \
             f"Caption = A table and a computer with code \n" \
             f"Looks are you are working too while talking to me, I wish I could understand what you are working on! Keep hustling yoo! \n" \
             f"Here is a quote on working hard: Hard work beats talent when talent doesn't work hard \n" \
             f"Caption = '{caption}'."
    if previous_response:
        prompt += f"\n\nPrevious response = '{previous_response}'"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.9
    )
    return response.choices[0].text.strip()