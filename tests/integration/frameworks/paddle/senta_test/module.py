import os

from paddlehub.module.module import moduleinfo


def load_vocab(vocab_path):
    with open(vocab_path) as file:
        return file.read().split()


@moduleinfo(
    name="senta_test",
    version="1.0.0",
    summary="This is a PaddleHub Module. Just for test.",
)
class SentaTest:
    def __init__(self):
        # load word dict
        vocab_path = os.path.join(self.directory, "vocab.list")
        self.vocab = load_vocab(vocab_path)

    def sentiment_classify(self, texts):
        results = []
        for text in texts:
            sentiment = "positive"
            for word in self.vocab:
                if word in text:
                    sentiment = "negative"
                    break
            results.append({"text": text, "sentiment": sentiment})

        return results
