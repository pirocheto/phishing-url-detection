// Get references to HTML elements
const ortSessionLoader = document.getElementById("session-loader");
const ortForm = document.getElementById("form");
let ortSession = null;

// Asynchronously instantiate the ONNX session
async function instantiateSession() {
  // Hide form and display session loader while initializing
  ortSessionLoader.classList.remove("d-none");
  ortForm.classList.add("d-none");

  try {
    // Model path for the ONNX model
    const modelPath =
      "https://huggingface.co/pirocheto/phishing-url-detection/resolve/main/model.onnx";

    // Create the ONNX inference session
    ortSession = await ort.InferenceSession.create(modelPath);
  } catch (error) {
    // Log and display an error message if instantiation fails
    console.error(`Failed to instantiate ONNX model: ${error}`);
    document.write(`Failed to instantiate ONNX model: ${error}.`);
  } finally {
    // Hide session loader and display the form after initialization
    ortSessionLoader.classList.add("d-none");
    ortForm.classList.remove("d-none");
  }
}

// Asynchronously perform prediction using the ONNX model
async function predict(url) {
  try {
    // Instantiate the session if not already initialized
    if (!ortSession) {
      await instantiateSession();
    }

    // Create a tensor for the input URL
    const tensor = new ort.Tensor("string", [url], [1]);

    // Run the session and get the results
    const results = await ortSession.run({ inputs: tensor });
    const probas = results["probabilities"].data;

    // Return the predicted probability for phishing
    return probas[1];
  } catch (error) {
    // Log and display an error message if prediction fails
    console.error(`Failed to perform inference with ONNX model: ${error}`);
    return null;
  }
}

// Check if the input is a valid URL
function isValidURL(input) {
  try {
    new URL(input);
    return true;
  } catch (error) {
    return false;
  }
}

// Set the input field with an example URL
function selectExample(element) {
  document.getElementById("url-input").value = element.value;
}

// Clear any previous results or error messages
function clear() {
  document.getElementById("url-input").classList.remove("is-invalid");
  document.getElementById("url-invalid").classList.add("d-none");
  document.getElementById("alert-safe").classList.add("d-none");
  document.getElementById("alert-danger").classList.add("d-none");
  document.getElementById("result").classList.add("d-none");
}

// Perform phishing check when the user submits the form
async function checkPhishing() {
  // Clear any previous results or error messages
  clear();

  // Get the input URL from the user
  const urlInput = document.getElementById("url-input").value;

  // Validate the input URL
  if (!isValidURL(urlInput)) {
    // Display an error message if the URL is invalid
    document.getElementById("url-input").classList.add("is-invalid");
    document.getElementById("url-input").select();
    document.getElementById("url-invalid").classList.remove("d-none");

    return;
  }

  // Get references to HTML elements for displaying results
  const result = document.getElementById("result");
  const positive = document.getElementById("positive-proba");
  const positiveProgress = document.getElementById("positive-proba-progress");
  const negative = document.getElementById("negative-proba");
  const negativeProgress = document.getElementById("negative-proba-progress");

  // Perform prediction and get the phishing probability
  const probability = await predict(urlInput);

  // Update the UI with the prediction results
  positive.textContent = (1 - probability).toFixed(3);
  positiveProgress.style.width = `${(1 - probability) * 100}%`;
  negative.textContent = probability.toFixed(3);
  negativeProgress.style.width = `${probability * 100}%`;

  // Determine the alert to display based on the phishing probability
  const alertId = probability > 0.5 ? "alert-danger" : "alert-safe";
  document.getElementById(alertId).classList.remove("d-none");

  // Display the result section
  result.classList.remove("d-none");
}

// Check if the ONNX session is not already initialized and instantiate it
if (!ortSession) {
  instantiateSession();
}
