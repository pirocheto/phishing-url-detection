const ort = require('onnxruntime-node');

const data = [
    {
        "url": "http://rapidpaws.com/wp-content/we_transfer/index2.php?email=/",
        "nb_hyperlinks": 1,
        "ratio_intHyperlinks": 1,
        "ratio_extHyperlinks": 0,
        "ratio_extRedirection": 0,
        "safe_anchor": 0,
        "domain_registration_length": 338,
        "domain_age": 0,
        "web_traffic":1853,
        "google_index": 1,
        "page_rank": 2,
    },
];

async function main() {
    try {
        // Make sure you have downloaded the model.onnx
        // Creating an ONNX inference session with the specified model
        const model_path = "./models/model.onnx";
        const session = await ort.InferenceSession.create(model_path);
        
        // Creating an ONNX tensor from the input data
        const inputs = data.map(url => Object.values(url).slice(1));
        const flattenInputs = inputs.flat();
        const tensor = new ort.Tensor('float32', flattenInputs, [inputs.length, 10]);
        
        // Executing the inference session with the input tensor
        const results = await session.run({"X": tensor});
        const probas = results['probabilities'].data;
        
        // Displaying results for each URL
        data.forEach((url, index) => {
            const proba = probas[index * 2 + 1];
            const percent = (proba * 100).toFixed(2);
            
            console.log(`URL: ${url.url}`);
            console.log(`Likelihood of being a phishing site: ${percent}%`);
            console.log("----");
        });
        

    } catch (e) {
        console.log(`failed to inference ONNX model: ${e}.`);
    }
};

main();

// Expected output:
// URL: https://www.rga.com/about/workplace
// Likelihood of being a phishing site: 0.89%
// ----
