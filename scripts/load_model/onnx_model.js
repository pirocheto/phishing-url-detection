const ort = require('onnxruntime-node');

const urls = ["https://www.rga.com/about/workplace", "https://hd1xor4.com/rd/c44979XCTyQ347384YbMC5057DBz8924Sclt660"];

async function main() {
    try {
        // Make sure you have downloaded the model.onnx
        // Creating an ONNX inference session with the specified model
        const model_path = "./models/model.onnx";
        const session = await ort.InferenceSession.create(model_path);
        
        // Executing the inference session with the input tensor
        const tensor = new ort.Tensor('string', urls);
        const results = await session.run({"X": tensor});
        const probas = results['probabilities'].data;

        // Displaying results for each URL
        urls.forEach((url, index) => {
            const proba = probas[index * 2 + 1];
            const percent = (proba * 100).toFixed(2);
            
            console.log(`URL: ${url}`);
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
// Likelihood of being a phishing site: 0.25%
// ----
