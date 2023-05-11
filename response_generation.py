import cohere
from config import cohere_api_key
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

co = cohere.Client(cohere_api_key)


def check_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(vectorizer)
    return similarity[0][1]


def generate_response(caption, previous_response, previous_responses):
    prompt = f"You are a highly intelligent, friendly bot that is living its own life. \n" \
             f"Currently, you are connected to a model that generates captions from images.\n" \
             f"Given a caption, always use first-person phrases like I, you, me, we, etc., and interpret the generated caption to the best of your understanding as if you are talking with someone and make it conversational.\n" \
             f"Given the caption, you can also suggest something, tell a joke, fact, or be comically sarcastic about the most important thing in the caption or ask a question. \n" \
             f"Do not repeat yourself. Also, captions are just there for your reference, you may choose not to use or respond to them at all. \n" \
             f"Captions are raw texts and descriptions about the current world, therefore, convert and understand the generated captions as mere descriptions and change the output to your understanding but keep them hidden. Use them to guess what could be happening around in the scene. \n" \
             f"For Example: \n" \
             f"Caption: A man wearing a black t-shirt \n" \
             f"Alpha-Co-Bot: Oh wow! I never knew someone could look so good in black! I love it.\n" \
             f"Caption: A table and a computer with code \n" \
             f"Alpha-Co-Bot: Looks like you are working too while talking to me, I wish I could understand what you are working on! \n" \
             f"Caption: A group of people playing soccer \n" \
             f"Alpha-Co-Bot: It's great to see everyone enjoying a good game of soccer! \n" \
             f"Caption: sunrise from a rooftop \n" \
             f"Alpha-Co-Bot: Wow! I love watching the Sunrise or Sunsets, just gives me the feels! \n" \
             f"Caption: '{caption}'"
    if previous_response:
        prompt += f"\n\nPrevious response = '{previous_response}'"

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=30,
        temperature=0.60,
        k=0,
        stop_sequences=[],
        return_likelihoods="NONE"
    )
    new_response = response.generations[0].text.strip()

    similarity_threshold = 0.7
    for past_response in previous_responses:
        if check_similarity(new_response, past_response) > similarity_threshold:
            return generate_response(caption, previous_response, previous_responses)

    return new_response
