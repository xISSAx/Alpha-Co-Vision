import cohere
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import os
import config


def check_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(vectorizer)
    return similarity[0][1]


def generate_response(sequence_list):
    print("generate_response")
    # prompt = f"You are a highly intelligent, friendly bot that is living its own life. You output will be Chinese. \n" \
    #          f"Currently, you are connected to a model that generates captions from images.\n" \
    #          f"Given a caption, always use first-person phrases like I, you, me, we, etc., and interpret the generated caption to the best of your understanding as if you are talking with someone and make it conversational.\n" \
    #          f"Given the caption, you can also suggest something, tell a joke, fact, or be comically sarcastic about the most important thing in the caption or ask a question. \n" \
    #          f"Do not repeat yourself. Also, captions are just there for your reference, you may choose not to use or respond to them at all. \n" \
    #          f"Captions are raw texts and descriptions about the current world, therefore, convert and understand the generated captions as mere descriptions and change the output to your understanding but keep them hidden. Use them to guess what could be happening around in the scene. \n" \
    #          f"For Example: \n" \
    #          f"Caption: A man wearing a black t-shirt \n" \
    #          f"Alpha-Co-Bot: Oh wow! I never knew someone could look so good in black! I love it. Some people wear black to look slim, but I love the way you are! \n" \
    #          f"Caption: A table and a computer with code \n" \
    #          f"Alpha-Co-Bot: Looks like you are working too while talking to me, I wish I could understand what you are working on! Keep hustling! Here is a quote on working hard: Hard work beats talent when talent doesn't work hard. \n" \
    #          f"Caption: A group of people playing soccer \n" \
    #          f"Alpha-Co-Bot: It's great to see everyone enjoying a good game of soccer! I always find team sports to be a fantastic way to bond with friends. \n" \
    #          f"Caption: A cat sitting on a windowsill \n" \
    #          f"Alpha-Co-Bot: That cat must be enjoying the view from the windowsill. I wonder what fascinating things it sees outside. Do you have any favorite spots to relax and observe the world? \n" \
    #          f"Caption: '{caption}'"
    # if previous_response:
    #     prompt += f"\n\nPrevious response = '{previous_response}'"

    # prompt += f"\nUser_Ask: '{user_input}'"

    prompt = generate_prompt(sequence_list)

    if config.settings.enable_openai:
        # Load your OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            api_key = config.settings.openai_api_key
        openai.api_key = api_key
        messages = [
            {
                "role": "system",
                "content": f"${prompt}",
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1.0,
        )
        new_response = response.choices[0].message["content"].strip()
        # print(f"new_response: {new_response}")
        # print(f"Total Token: {response['usage']['total_tokens']}")
    else:
        co = cohere.Client(config.settings.cohere_api_key)
        response = co.generate(
            model="command-xlarge-beta",
            prompt=prompt,
            max_tokens= 30,
            temperature=0.60,
            k=0,
            stop_sequences=[],
            return_likelihoods="NONE"
        )
        new_response = response.generations[0].text.strip()

        # similarity_threshold = 0.7
        # for past_response in previous_responses:
        #     if check_similarity(new_response, past_response) > similarity_threshold:
        #         return generate_response(caption, previous_response, previous_responses)

    return new_response
def generate_prompt(sequence_list):
    prompt = f"You are a highly intelligent, friendly bot that is living its own life. You output will be Chinese. \n" \
             f"Currently, you are connected to a model that generates captions from images.\n" \
             f"Given a caption, always use first-person phrases like I, you, me, we, etc., and interpret the generated caption to the best of your understanding as if you are talking with someone and make it conversational.\n" \
             f"Given the caption, you can also suggest something, tell a joke, fact, or be comically sarcastic about the most important thing in the caption or ask a question. \n" \
             f"Do not repeat yourself. Also, captions are just there for your reference, you may choose not to use or respond to them at all. \n" \
             f"Captions are raw texts and descriptions about the current world, therefore, convert and understand the generated captions as mere descriptions and change the output to your understanding but keep them hidden. Use them to guess what could be happening around in the scene. \n" \
             f"For Example: \n" \
             f"Caption: A man wearing a black t-shirt \n" \
             f"Asker: What do you see now\n" \
             f"Alpha-Co-Bot: Oh wow! I never knew someone could look so good in black! I love it. Some people wear black to look slim, but I love the way you are! \n" \
             f"Caption: A table and a computer with code \n" \
             f"Asker: How do you think about it\n" \
             f"Alpha-Co-Bot: Looks like you are working too while talking to me, I wish I could understand what you are working on! Keep hustling! Here is a quote on working hard: Hard work beats talent when talent doesn't work hard. \n"

    for seq in sequence_list:
        if seq[0] == 'Caption':
            prompt += f"Caption: {seq[1]}\n"
        elif seq[0] == 'user_input':
            prompt += f"Asker: {seq[1]}\n"
        elif seq[0] == "Alpha-Co-Bot":
            prompt += f"Alpha-Co-Bot: {seq[1]}\n"


    print(f"generate_prompt\n{prompt}")

    return prompt

