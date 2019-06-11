import responder

from .livedoor_news import parse_japanese, get_classifications, get_tokenizer
from .model import load_model

api = responder.API(cors=True)
api.add_route("/", static=True)


@api.on_event("startup")
async def startup():
    api.model = load_model()
    api.tokenizer = get_tokenizer()


@api.route("/scores")
async def get_scores(req, resp):
    req_body = await req.media()
    input_text = req_body["text"]
    parsed_text = parse_japanese(input_text)

    scores = api.model.predict(
        api.tokenizer.texts_to_matrix([parsed_text], mode="tfidf")
    )
    descriptions = map(lambda x: x[1], get_classifications())

    resp.media = {
        "inputTest": input_text,
        "scores": sorted(
            [
                {"description": description, "value": float(value)}
                for description, value in zip(descriptions, scores[0])
            ],
            key=lambda x: x["value"],
            reverse=True,
        ),
    }
