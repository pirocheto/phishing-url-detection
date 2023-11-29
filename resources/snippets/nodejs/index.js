const ort = require('onnxruntime-node');

async function main() {
    
    try {
        // Make sure you have downloaded the model.onnx
        // Creating an ONNX inference session with the specified model
        const model_path = "./model.onnx";
        const session = await ort.InferenceSession.create(model_path);

        const urls = [
            "https://clubedemilhagem.com/home.php",
            "http://www.medicalnewstoday.com/articles/188939.php",
        ]
        
        // Creating an ONNX tensor from the input data
        const tensor = new ort.Tensor('string', urls, [urls.length,]);
        
        // Executing the inference session with the input tensor
        const results = await session.run({"inputs": tensor});
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