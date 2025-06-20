/**
 * frontend.js
 * Provides image rendering and canvas painting functionality.
 * Import this file in your HTML with: <script src="frontend.js"></script>
 */

// Image filenames for each column
const imageDescriptions = [
    'Quantum Neural Network Image 1',
    'Quantum Neural Network Image 2',
];
const quantumImages = [
    'images/quantum1.png',
    'images/quantum2.png',
];

const hybridImages = [
    'images/hybrid1.png',
    'images/hybrid3.png',
];

const classicalImages = [
    'images/classical01.png',
    'images/classical2.png',
];

var prediction = '';

function renderImages(containerId, images) {
    const container = document.getElementById(containerId);
    if (!container) return;
    images.forEach(src => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = '';
        container.appendChild(img);
    });
}

// Call these after DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    renderImages('quantum-images', quantumImages);
    renderImages('hybrid-images', hybridImages);
    renderImages('classical-images', classicalImages);

    // Painting functionality
    const canvas = document.getElementById("canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    // Set background to black
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    let painting = false;

    canvas.addEventListener("mousedown", () => (painting = true));
    canvas.addEventListener("mouseup", () => (painting = false));
    canvas.addEventListener("mouseleave", () => (painting = false));
    canvas.addEventListener("mousemove", (event) => {
        if (!painting) return;
        ctx.fillStyle = "white"; // Draw in white
        ctx.fillRect(event.offsetX, event.offsetY, 10, 10);
    });

    // Expose getImage to global scope for button onclick, etc.
    window.getImage = function getImage() {
        // Resize the image to 28x28 pixels
        const smallCanvas = document.createElement("canvas");
        smallCanvas.width = 28;
        smallCanvas.height = 28;
        const smallCtx = smallCanvas.getContext("2d");
        smallCtx.fillStyle = "black";
        smallCtx.fillRect(0, 0, 28, 28);
        smallCtx.drawImage(canvas, 0, 0, 28, 28);
        const imageData = smallCanvas.toDataURL("image/png");
        // Create a download link
        const link = document.createElement("a");
        link.href = imageData;
        link.download = "drawing.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        // Display or store the image
        console.log(imageData);
    };

    // Add an event listener to the button to upload the image at http://127.0.0.1:5000/upload
    // (Button should exist in your HTML with id="uploadBtn")
    const uploadBtnHybrid = document.getElementById("uploadBtnHybrid");
    if (uploadBtnHybrid) {
        uploadBtnHybrid.addEventListener("click", async () => {
            // Get the 28x28 image data
            const smallCanvas = document.createElement("canvas");
            smallCanvas.width = 28;
            smallCanvas.height = 28;
            const smallCtx = smallCanvas.getContext("2d");
            smallCtx.fillStyle = "black";
            smallCtx.fillRect(0, 0, 28, 28);
            smallCtx.drawImage(canvas, 0, 0, 28, 28);
            const blob = await new Promise(resolve => smallCanvas.toBlob(resolve, "image/png"));
            const formData = new FormData();
            formData.append("file", blob, "drawing.png");
            try {
                const response = await fetch("http://127.0.0.1:5000/upload_and_predict_hybrid", {
                    method: "POST",
                    body: formData
                });
                if (response.ok) {
                    // Handle successful upload
                    const result = await response.json();
                    alert("Image uploaded successfully!");
                    console.log("Server response:", result);
                    // Optionally, you can display the result or do something with it
                    console.log(result);
                    //console.log("Message:", result.message);
                    console.log("Label:", result.label);
                    prediction = result.label;
                    console.log(" PREDICTION: ", prediction);
                    // Call this function after prediction is set (e.g., after upload)
                    predictionText.textContent = `${prediction}`;
                    //window.updatePredictionText = updatePredictionText;
                    // if (result) {
                    //     alert("Predicted label for the provided image is: " + result.label);
                    // } else {
                    //     alert("Upload successful, but no message returned.");
                    // }
                    alert("Predicted label for the provided image is: " + result.label);
                } else {
                    alert("Upload failed.");
                }
            } catch (err) {
                alert("Error uploading image.");
            }
        });
    }

    const uploadBtnClassical = document.getElementById("uploadBtnClassical");
    if (uploadBtnClassical) {
        uploadBtnClassical.addEventListener("click", async () => {
            // Get the 28x28 image data
            const smallCanvas = document.createElement("canvas");
            smallCanvas.width = 28;
            smallCanvas.height = 28;
            const smallCtx = smallCanvas.getContext("2d");
            smallCtx.fillStyle = "black";
            smallCtx.fillRect(0, 0, 28, 28);
            smallCtx.drawImage(canvas, 0, 0, 28, 28);
            const blob = await new Promise(resolve => smallCanvas.toBlob(resolve, "image/png"));
            const formData = new FormData();
            formData.append("file", blob, "drawing.png");
            try {
                const response = await fetch("http://127.0.0.1:5000/upload_and_predict_cnn", {
                    method: "POST",
                    body: formData
                });
                if (response.ok) {
                    // Handle successful upload
                    const result = await response.json();
                    alert("Image uploaded successfully!");
                    console.log("Server response:", result);
                    // Optionally, you can display the result or do something with it
                    console.log(result);
                    //console.log("Message:", result.message);
                    console.log("Label:", result.label);
                    prediction = result.label;
                    predictionText.textContent = `${prediction}`;
                    alert("Predicted label for the provided image is: " + result.label);
                } else {
                    alert("Upload failed.");
                }
            } catch (err) {
                alert("Error uploading image.");
            }
        });
    }

    // Add an event listener to the button to clear the canvas
    const clearBtn = document.getElementById("clearCanvas");
    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            ctx.fillStyle = "black"; // Clear to black
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        });
    }
    console.log(" PREDICTION FIRST: ", prediction);
    // Put the prediction in the HTML element with id="predictionText"
    async function updatePredictionText() {
        const predictionText = document.getElementById("predictionText");
        // Wait until prediction is updated (not empty string)
        while (prediction === '') {
            await new Promise(resolve => setTimeout(resolve, 100));
            console.log(" PREDICTION 1: ", prediction);
        }
        predictionText.textContent = `${prediction}`;
        console.log(" PREDICTION 2: ", prediction);
    }

    

    
});