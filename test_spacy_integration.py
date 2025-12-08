import os
import sys
from typing import Any

import spacy
from spacy.tokens import Doc

import bentoml

# Add the project root to Python path to import our implementation
sys.path.insert(0, os.path.abspath("."))

# Import our spaCy implementation
from src._bentoml_impl.frameworks import spacy as bentoml_spacy


# Create a simple spaCy model
def create_test_model() -> Any:  # Using Any instead of Language to avoid linter issues
    # Create a blank English model
    nlp = spacy.blank("en")

    # Add a simple component
    @nlp.component("print_tokens")  # Using nlp.component instead of spacy.Language.component
    def print_tokens(doc: Doc) -> Doc:
        for token in doc:
            print(token.text)
        return doc

    nlp.add_pipe("print_tokens")
    return nlp

# Test saving and loading
def test_save_load() -> bentoml.Model:
    model = create_test_model()

    # Save the model
    saved_model = bentoml_spacy.save("spacy_test_model", model)
    print(f"Model saved: {saved_model.tag}")

    # Load the model
    loaded_model = bentoml_spacy.load(saved_model.tag)

    # Verify the loaded model works
    doc = loaded_model("Hello world")
    assert len(doc) == 2
    print("Save and load test passed!")
    return saved_model

# Test runnable
def test_runnable(model_tag: str) -> None:
    # Get runnable class
    runnable_cls = bentoml_spacy.get_runnable(bentoml.models.get(model_tag))

    # Create runnable instance
    runner = runnable_cls()

    # Test __call__
    result = runner("Testing the SpaCy runnable")
    assert len(result) == 4

    # Test pipe
    texts = ["First document", "Second document", "Third document"]
    results = list(runner.pipe(texts, batch_size=2))
    assert len(results) == 3

    print("Runnable test passed!")

if __name__ == "__main__":
    # Run tests
    saved_model = test_save_load()
    test_runnable(saved_model.tag)
    print("All tests passed successfully!")
