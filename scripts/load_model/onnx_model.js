const ort = require('onnxruntime-node');

const urls = [
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

        // Creating an ONNX inference session with the specified model
        const model_path = "./models/model.onnx";
        const session = await ort.InferenceSession.create(model_path);
        
        // Get values from data and remove url links
        const inputs = urls.map(url => Object.values(url).slice(1));
        
        // Flattening the 2D array to get a 1D array
        const flattenInputs = inputs.flat();
        
        // Creating an ONNX tensor from the input array
        const tensor = new ort.Tensor('float32', flattenInputs, [inputs.length, 10]);
        
        // Executing the inference session with the input tensor
        const results = await session.run({"X": tensor});
        
        // Retrieving probability data from the results
        const probas = results['probabilities'].data;
        
        // Displaying results for each URL
        urls.forEach((url, index) => {
            // The index * 2 + 1 is used to access the probability associated with the phishing class
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

// output:
// URL: https://www.rga.com/about/workplace
// Likelihood of being a phishing site: 0.89%
// ----
