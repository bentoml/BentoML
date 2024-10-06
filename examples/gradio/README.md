# Quickstart

This quickstart demonstrates how to add Gradio web UI to a BentoML service.

## Prerequisites

Python 3.9+ and `pip` installed. See the [Python downloads page](https://www.python.org/downloads/) to learn more.

## Get started

Perform the following steps to run this project and deploy it to BentoCloud.

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Serve your model as an HTTP server. This starts a local server at [http://localhost:3000](http://localhost:3000/), making your model accessible as a web service.

   ```bash
   bentoml serve .
   ```

3. Visit http://localhost:3000/ui for gradio UI. BentoML APIs can be found at http://localhost:3000
