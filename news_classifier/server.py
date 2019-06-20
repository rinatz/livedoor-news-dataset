from marshmallow import Schema, fields
import responder

from .livedoor_news import tokenize_japanese, get_classifications, get_tokenizer
from .model import load_model

api = responder.API(
    title="REST API",
    description="Created by Responder",
    openapi="3.0.2",
    docs_route="/docs",
    cors=True,
    cors_params={
        "allow_origins": ["*"],
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    },
)
api.add_route("/", static=True)


@api.schema("ClassificationRequest")
class ClassificationRequestSchema(Schema):
    text = fields.Str(required=True, description="分析するテキスト")


@api.schema("Prediction")
class PredictionSchema(Schema):
    description = fields.Str(required=True, description="分類項目")
    possibility = fields.Float(required=True, description="その項目に分類される可能性")


@api.schema("TFIDF")
class TFIDFSchema(Schema):
    word = fields.Str(required=True, description="入力テキストのトークン化されたワード")
    value = fields.Float(required=True, description="TF-IDF の値")


@api.schema("ClassificationResponse")
class ClassificationResponseSchema(Schema):
    inputTest = fields.Str(required=True, description="入力テキスト")
    predictions = fields.Nested(
        PredictionSchema, required=True, many=True, description="分類結果"
    )
    tfidf = fields.Nested(
        TFIDFSchema, required=True, many=True, description="各ワードの TF-IDF"
    )


@api.on_event("startup")
async def startup():
    api.model = load_model()
    api.tokenizer = get_tokenizer()


@api.route("/classification")
async def classification(req, resp):
    """テキストのトピック分析を行います。
    ---
    post:
      tags:
        - 分類器
      summary: テキストのトピック分析を行います。
      description: 入力されたテキストのトピック分析の結果と TF-IDF を取得します。
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ClassificationRequest"
      responses:
        "200":
          description: 成功
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ClassificationResponse"
    """

    req_body = await req.media()
    input_text = req_body["text"]
    tokenized_text = tokenize_japanese(input_text)
    texts = [" ".join(tokenized_text)]
    tfidf = api.tokenizer.texts_to_matrix(texts, mode="tfidf")

    possibilities = api.model.predict(tfidf)
    descriptions = map(lambda x: x[1], get_classifications())

    word_tfidf = {}

    for word in tokenized_text:
        try:
            index = api.tokenizer.word_index[word]
            word_tfidf[word] = tfidf[0][index]
        except KeyError:
            word_tfidf[word] = 0.0

    resp.media = ClassificationResponseSchema().dump(
        {
            "inputText": input_text,
            "predictions": sorted(
                [
                    {"description": description, "possibility": possibility}
                    for description, possibility in zip(descriptions, possibilities[0])
                ],
                key=lambda x: x["possibility"],
                reverse=True,
            ),
            "tfidf": sorted(
                [{"word": word, "value": value} for word, value in word_tfidf.items()],
                key=lambda x: x["value"],
                reverse=True,
            ),
        }
    )
