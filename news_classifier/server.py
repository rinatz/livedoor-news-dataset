import responder

from .livedoor_news import tokenize_japanese, get_classifications, get_tokenizer
from .model import load_model

api = responder.API(
    cors=True,
    cors_params={
        "allow_origins": ["*"],
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    },
)
api.add_route("/", static=True)


@api.on_event("startup")
async def startup():
    api.model = load_model()
    api.tokenizer = get_tokenizer()


@api.route("/scores")
async def get_scores(req, resp):
    req_body = await req.media()
    input_text = req_body["text"]
    tokenized_text = tokenize_japanese(input_text)
    texts = [" ".join(tokenized_text)]
    tfidf = api.tokenizer.texts_to_matrix(texts, mode="tfidf")

    scores = api.model.predict(tfidf)
    descriptions = map(lambda x: x[1], get_classifications())

    word_tfidf = {}

    for word in tokenized_text:
        try:
            index = api.tokenizer.word_index[word]
            word_tfidf[word] = tfidf[0][index]
        except KeyError:
            word_tfidf[word] = 0.0

    resp.media = {
        "inputText": input_text,
        "classification": sorted(
            [
                {"description": description, "score": float(score)}
                for description, score in zip(descriptions, scores[0])
            ],
            key=lambda x: x["score"],
            reverse=True,
        ),
        "tfidf": sorted(
            [
                {"word": word, "value": value}
                for word, value in word_tfidf.items()
            ],
            key=lambda x: x["value"],
            reverse=True,
        ),
    }
