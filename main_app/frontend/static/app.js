const form = document.getElementById('predictForm');
const resultText = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const sampleID = document.getElementById('Sample_ID').value;
    const features = [
        parseFloat(document.getElementById('Moisture').value),
        parseFloat(document.getElementById('Ash').value),
        parseFloat(document.getElementById('Volatile_Oil').value),
        parseFloat(document.getElementById('Acid_Insoluble_Ash').value),
        parseFloat(document.getElementById('Chromium').value),
        parseFloat(document.getElementById('Coumarin').value),
        parseFloat(document.getElementById('Fiber').value),
        parseFloat(document.getElementById('Density').value),
        parseFloat(document.getElementById('Oil_Content').value),
        parseFloat(document.getElementById('Resin').value),
        parseFloat(document.getElementById('Pesticide_Level').value),
        parseFloat(document.getElementById('PH_Value').value)
    ];

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ Sample_ID: sampleID, features })
        });

        const data = await response.json();
        console.log(data);

        if (data.prediction) {
            resultText.innerText = `Sample ${data.Sample_ID} → Predicted Quality: ${data.prediction}`;
        } else {
            resultText.innerText = `Error: ${data.error}`;
        }

    } catch (err) {
        resultText.innerText = 'Error calling API';
        console.error(err);
    }
});
