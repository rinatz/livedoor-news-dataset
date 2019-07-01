from marshmallow import Schema, fields
import responder

from .japanese import MeCabTokenizer
from .livedoor_news import get_classifications, get_tokenizer
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


@api.schema("Document")
class DocumentSchema(Schema):
    text = fields.Str(required=True, description="分析するテキスト")


@api.schema("ClassificationCategory")
class ClassificationCategorySchema(Schema):
    name = fields.Str(required=True, description="分類項目")
    confidence = fields.Float(required=True, description="分類結果に対する信頼性")


@api.schema("Token")
class TokenSchema(Schema):
    lemma = fields.Str(required=True, description="入力テキストに含まれる単語の見出し語")
    tfidf = fields.Float(required=True, description="単語の tf-idf")


@api.schema("Classification")
class ClassificationSchema(Schema):
    text = fields.Str(required=True, description="入力テキスト")
    categories = fields.Nested(
        ClassificationCategorySchema, required=True, many=True, description="分類結果"
    )
    tokens = fields.Nested(
        TokenSchema, required=True, many=True, description="テキストをトークン化した結果"
    )


@api.on_event("startup")
async def startup():
    api.model = load_model()
    api.tokenizer = get_tokenizer()
    api.mecab = MeCabTokenizer()


@api.route("/classifications")
async def classifications(req, resp):
    """テキストのトピック分析を行います。
    ---
    post:
      tags:
        - 分類器
      summary: テキストのトピック分析を行います。
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/Document"
      responses:
        "200":
          description: 成功
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Classification"
    """

    req_body = await req.media()
    input_text = req_body["text"]
    tokenized_text = api.mecab.tokenize(input_text)
    texts = [" ".join(tokenized_text)]
    tfidf = api.tokenizer.texts_to_matrix(texts, mode="tfidf")

    confidences = api.model.predict(tfidf)
    names = get_classifications().values()

    word_tfidf = {}

    for word in tokenized_text:
        try:
            index = api.tokenizer.word_index[word]
            word_tfidf[word] = tfidf[0][index]
        except KeyError:
            word_tfidf[word] = 0.0

    resp.media = ClassificationSchema().dump(
        {
            "text": input_text,
            "categories": sorted(
                [
                    {"name": name, "confidence": confidence}
                    for name, confidence in zip(names, confidences[0])
                ],
                key=lambda x: x["confidence"],
                reverse=True,
            ),
            "tokens": sorted(
                [{"lemma": word, "tfidf": tfidf} for word, tfidf in word_tfidf.items()],
                key=lambda x: x["tfidf"],
                reverse=True,
            ),
        }
    )
