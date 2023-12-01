let ortSession = null;

async function instantiateSess() {
  try {
    const model_path =
      "https://huggingface.co/pirocheto/phishing-url-detection/resolve/main/model.onnx";
    ortSession = await ort.InferenceSession.create(model_path);
  } catch (e) {
    document.write(`failed to instantiate ONNX model: ${e}.`);
  }
}

async function predict(url) {
  try {
    if (!ortSession) {
      await instantiateSess();
    }

    const tensor = new ort.Tensor("string", [url], [1]);
    const results = await ortSession.run({ inputs: tensor });
    const probas = results["probabilities"].data;
    return probas[1];
  } catch (e) {
    document.write(`failed to perform inference with ONNX model: ${e}.`);
  }
}

function isValidURL(input) {
  try {
    new URL(input);
    return true;
  } catch (error) {
    return false;
  }
}

async function checkPhishing() {
  var urlInput = document.getElementById("urlInput").value;
  var resultElement = document.getElementById("result");

  // Validate the input URL
  if (!isValidURL(urlInput)) {
    resultElement.innerText = "Invalid URL. Please enter a valid URL.";
    return;
  }

  // Perform the phishing check
  var probability = await predict(urlInput);

  // Display the probability
  resultElement.innerText = "Phishing Probability: " + probability.toFixed(2);
}
