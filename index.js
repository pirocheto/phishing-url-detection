let ortSession = null;

async function instantiateSess() {
  document.getElementById("session-loader").style.display = "flex";
  try {
    const model_path =
      "https://huggingface.co/pirocheto/phishing-url-detection/resolve/main/model.onnx";
    ortSession = await ort.InferenceSession.create(model_path);
  } catch (e) {
    document.write(`failed to instantiate ONNX model: ${e}.`);
  } finally {
    document.getElementById("session-loader").style.display = "none";
  }
}

async function predict(url) {
  document.getElementById("check-button").disabled = true;
  document.getElementById("check-button").classList.add("opacity-50");
  document.getElementById("check-loader").style.display = "flex";
  document.getElementById("check-text").style.display = "none";

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
    return null;
  } finally {
    document.getElementById("check-loader").style.display = "none";
    document.getElementById("check-button").classList.remove("opacity-50");
    document.getElementById("check-text").style.display = "flex";
    document.getElementById("check-button").disabled = false;
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
  var urlElement = document.getElementById("url");
  var scoreElement = document.getElementById("score");
  var messageElement = document.getElementById("message");

  // Validate the input URL
  if (!isValidURL(urlInput)) {
    resultElement.innerText = "Invalid URL. Please enter a valid URL.";
    return;
  }

  // Perform the phishing check
  var probability = await predict(urlInput);

  const warningMessage = `
    <span class="flex items-center">
      can be dangerous
      <img src="assets/warning.svg" class="h-4 w-4 ml-2" /> 
    </span>`;

  const safeMessage = `
    <span class="flex items-center">
      seems to be safe
      <img src="assets/safe.svg" class="h-4 w-4 ml-2" /> 
    </span>`;

  const message = probability > 0.5 ? safeMessage : warningMessage;

  urlElement.innerText = urlInput;
  scoreElement.innerText = probability.toFixed(2);
  messageElement.innerHTML = message;
  resultElement.style.display = "block";
}

if (!ortSession) {
  instantiateSess();
}
