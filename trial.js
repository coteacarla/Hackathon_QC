    const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let currentTab = 'overview';

        // Initialize canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';

        // Drawing functions
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // Touch events for mobile
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouch);
        canvas.addEventListener('touchend', stopDrawing);

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function draw(e) {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        function stopDrawing() {
            if (isDrawing) {
                isDrawing = false;
                ctx.beginPath();
            }
        }

        // Image popup functionality
        function showImagePopup(src, alt) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
            document.getElementById('modalCaption').textContent = alt;
        }

        function closeImagePopup() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal when pressing Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeImagePopup();
            }
        });

        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                            e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('predictionResult').textContent = 'Draw a digit!';
            document.getElementById('confidenceBars').innerHTML = '';
        }

        function saveImage() {
            const link = document.createElement('a');
            link.download = 'digit.png';
            link.href = canvas.toDataURL();
            link.click();
        }

        async function predictDigit() {
            // Predict with Classical NN
            await predictWithNetwork("classical", "http://127.0.0.1:5000/upload_and_predict_cnn");
            // Predict with Hybrid NN
            await predictWithNetwork("hybrid", "http://127.0.0.1:5000/upload_and_predict_hybrid");
            // Predict with Quantum NN
            //await predictWithNetwork("quantum", "http://127.0.0.1:5000/upload_and_predict_quantum");
        }

        async function predictWithNetwork(networkType, endpoint) {
            // Prepare 28x28 image data
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
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });
            if (response.ok) {
                const result = await response.json();
                // Display result if this network is the active tab
                if (
                (currentTab === "overview" && networkType === "hybrid") ||
                currentTab === networkType
                ) {
                document.getElementById('predictionResult').textContent = result.label;
                if (result.confidences) {
                    displayConfidenceBars(result.confidences);
                }
                }
            } else {
                alert(`Upload failed for ${networkType} network.`);
            }
            } catch (err) {
            alert(`Error uploading image for ${networkType} network.`);
            }
        }

        function simulateNetworkPredictions() {
            // Simulate different network behaviors
            const digit = Math.floor(Math.random() * 10);
            
            return {
                classical: generatePrediction(digit, 0.95, 'classical'),
                hybrid: generatePrediction(digit, 0.97, 'hybrid'),
                quantum: generatePrediction(digit, 0.85, 'quantum')
            };
        }

        function generatePrediction(correctDigit, baseAccuracy, networkType) {
            const confidences = Array(10).fill(0);
            
            // Add some realistic noise to predictions
            for (let i = 0; i < 10; i++) {
                if (i === correctDigit) {
                    confidences[i] = baseAccuracy + (Math.random() - 0.5) * 0.1;
                } else {
                    confidences[i] = Math.random() * (1 - baseAccuracy) / 9;
                }
            }
            
            // Normalize to sum to 1
            const sum = confidences.reduce((a, b) => a + b, 0);
            return confidences.map(c => c / sum);
        }

        function displayPredictionResults(results) {
            // Display prediction for current active tab
            const networkTypes = ['classical', 'hybrid', 'quantum'];
            const currentNetwork = currentTab === 'overview' ? 'hybrid' : currentTab;
            
            if (networkTypes.includes(currentNetwork)) {
                const prediction = results[currentNetwork];
                const predictedDigit = prediction.indexOf(Math.max(...prediction));
                
                document.getElementById('predictionResult').textContent = predictedDigit;
                displayConfidenceBars(prediction);
            }
        }

        function displayConfidenceBars(confidences) {
            const barsContainer = document.getElementById('confidenceBars');
            barsContainer.innerHTML = '';
            
            confidences.forEach((confidence, digit) => {
                const barElement = document.createElement('div');
                barElement.className = 'confidence-bar';
                barElement.innerHTML = `
                    <div class="digit-label">${digit}</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                    <div class="confidence-value">${(confidence * 100).toFixed(1)}%</div>
                `;
                barsContainer.appendChild(barElement);
            });
        }

        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            currentTab = tabName;
        }

        // Initialize with some sample confidence bars
        window.addEventListener('load', () => {
            const sampleConfidences = [0.05, 0.03, 0.08, 0.75, 0.02, 0.01, 0.03, 0.01, 0.01, 0.01];
            displayConfidenceBars(sampleConfidences);
            document.getElementById('predictionResult').textContent = '3';
        });