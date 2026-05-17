// Configuration
const API_BASE_URL = window.location.origin;
let isAnalyzing = false;
let currentResult = null;

// DOM Elements
var arrow = $(".arrow");
var input = $(".img-input");
var text = $(".link-input");
var upload = $(".upload");
var results = $(".results");
var results_text = $(".results-text");
var light = $(".light");
const imagePreview = document.getElementById('preview');

// Initialize
$(document).ready(function() {
    checkServerStatus();
    setupEventListeners();
});

function checkServerStatus() {
    fetch(`${API_BASE_URL}/health`)
        .then(response => response.json())
        .then(data => {
            console.log('Server status:', data);
        })
        .catch(error => {
            console.error('Cannot connect to server:', error);
        });
}

function setupEventListeners() {
    // File input change
    input.on('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
    
    // URL input
    input.on('input', function() {
        if (input.prop('type') === 'url') {
            const url = this.value.trim();
            if (url) {
                imagePreview.src = url;
                imagePreview.style.display = 'block';
            } else {
                imagePreview.src = '';
                imagePreview.style.display = 'none';
            }
        }
    });
}

// Mobile view handling
if (window.innerWidth < 701) {
    arrow.remove();
    results_text.addClass("link-input2");
    
    function color() {
        if (isAnalyzing) return false;
        if (input.val() === "" || (input.prop('type') === 'file' && !input[0].files.length)) {
            if (input.prop('type') == 'file') {
                alert("Please Upload Image First!");
                return false;
            } else {
                alert("Please Provide Link First!");
                return false;
            }
        }
        
        upload.css("animation", "");
        results.css("animation", "");
        upload.css("z-index", "0");
        upload[0].offsetHeight;
        results[0].offsetHeight;
        upload.css("animation", "flip-upload 1s 1 linear forwards");
        results.css("animation", "flip-results 1s 1 linear forwards");
        
        setTimeout(() => {
            analyzeImage();
        }, 1000);
    }
    
    function goBack() {
        results_text.css("z-index", "0");
        upload.css("z-index", "1");
        upload.css("animation", "");
        results.css("animation", "");
        upload[0].offsetHeight;
        results[0].offsetHeight;
        upload.css("animation", "flip-upload 1s 1 linear forwards reverse");
        results.css("animation", "flip-results 1s 1 linear forwards reverse");
        
        setTimeout(() => {
            resetUI();
        }, 1000);
    }
    
    results_text.on('click', goBack);
} else {
    function color() {
        if (isAnalyzing) return false;
        if (input.val() !== "") {
            arrow.css("animation", "arrowcol 1s 1 linear forwards");
            analyzeImage();
            light.css("display" , "none");
        results_text.removeClass("light-effect");
        } else {
            if (input.prop('type') == 'file') {
                alert("Please Upload Image First!");
            } else {
                alert("Please Provide Link First!");
            }
        }
    }
}

function analyzeImage() {
    if (isAnalyzing) return;
    
    const formData = new FormData();
    
    if (input.prop('type') === 'file' && input[0].files.length > 0) {
        // File upload
        const file = input[0].files[0];
        if (file.size > 16 * 1024 * 1024) {
            alert("File is too large! Maximum size is 16MB.");
            return;
        }
        formData.append('image', file);
    } else if (input.prop('type') === 'url' && input.val().trim() !== '') {
        // URL input
        formData.append('image_url', input.val().trim());
        
    } else {
        alert("Please provide an image first!");
        return;
    }
    
    isAnalyzing = true;
    showLoadingState();
    
    fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        isAnalyzing = false;
        
        if (data.success) {
            currentResult = data;
            displayResults(data);
        } else {
            showErrorState(data.error || "Analysis failed");
        }
    })
    .catch(error => {
        isAnalyzing = false;
        console.error('Error:', error);
        showErrorState("Connection error. Please check server.");
    });
}

function showLoadingState() {
    results_text.html(`
        <div style="text-align: center; padding: 30px;">
            <div class="spinner"></div>
            <div style="margin-top: 20px; font-size: 18px; color: #3498db;">
                Analyzing Image...
            </div>
            <div style="margin-top: 10px; font-size: 14px; color: #666;">
                Processing with AI ensemble
            </div>
        </div>
    `);
}

function showErrorState(errorMessage) {
    results_text.html(`
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; color: #e74c3c; margin-bottom: 15px;">
                ❌ Error
            </div>
            <div style="font-size: 16px; color: #666; margin-bottom: 20px;">
                ${errorMessage}
            </div>
            <button onclick="retryAnalysis()" 
                    style="padding: 10px 20px; background: #3498db; color: white; 
                           border: none; border-radius: 5px; cursor: pointer;">
                Try Again
            </button>
        </div>
    `);
    
    arrow.css("animation", "");
}

function retryAnalysis() {
    if (input.val() !== "") {
        analyzeImage();
    } else {
        resetUI();
    }
}

function displayResults(data) {
    const isFake = data.final_decision === 'FAKE';
    const fakeColor = '#e74c3c';
    const realColor = '#2ecc71';
    
    // FIXED: Use correct property names
    const finalConfidence = data.final_confidence || 0;
    const fakeProbability = data.fake_probability || 0;
    const realProbability = data.real_probability || 0;
    
    // Create clean UI
    let resultHTML = `
        <div style="height: 100%; overflow-y: auto; padding: 20px;margin-right: 70px">
    `;
    
    // 1. HEATMAP
    if (data.heatmap) {
        resultHTML += `
            <div style="text-align: center;animation: show-up 0.5s linear; margin-bottom: 25px;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50;">
                    Analysis Heatmap
                </div>
                <img src="data:image/png;base64,${data.heatmap}" 
                     style="max-width: 200px; max-height: 120px; border-radius: 10px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            </div>
        `;
    }
    
    // 2. FINAL DECISION (FIXED)
    resultHTML += `
        <div style="text-align: center;opacity: 0;animation: show-up 0.5s linear 0.7s forwards; margin-bottom: 25px; padding: 20px;
                    background: ${isFake ? 'rgba(231, 76, 60, 0.1)' : 'rgba(46, 204, 113, 0.1)'};
                    border-radius: 15px; border: 3px solid ${isFake ? fakeColor : realColor};">
            <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;
                        color: ${isFake ? fakeColor : realColor};">
                ${data.final_decision}
            </div>
            <div style="font-size: 24px; color: #2c3e50;">
                Confidence: <strong>${finalConfidence}%</strong>
            </div>
        </div>
    `;
    
    // 3. PROBABILITY BARS (FIXED)
    resultHTML += `
        <div style="margin-bottom: 25px;opacity: 0;animation: show-up 0.5s linear 1.2s forwards">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div style="font-weight: bold; color: ${fakeColor};">FAKE</div>
                <div style="font-weight: bold; color: ${realColor};">REAL</div>
            </div>
            <div style="height: 25px; background: #ecf0f1; border-radius: 12px; overflow: hidden;">
                <div style="width: ${fakeProbability}%; height: 100%; 
                            background: ${fakeColor}; float: left;"></div>
                <div style="width: ${realProbability}%; height: 100%; 
                            background: ${realColor}; float: left;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 14px;">
                <div>${fakeProbability}%</div>
                <div>${realProbability}%</div>
            </div>
        </div>
    `;
    
    // 4. VIEW DETAILS BUTTON - Now opens new page
    resultHTML += `
        <div style="margin-top: 30px;opacity: 0;animation: show-up 0.5s linear 1.7s forwards">
            <button onclick="viewDetailedAnalysis()" 
                    style="width: 100%; padding: 12px; background: #3498db; color: white;
                           border: none; border-radius: 8px; cursor: pointer;
                           font-size: 16px; font-weight: bold;">
                View Detailed Analysis
            </button>
        </div>
    `;
    
    // 5. NEW ANALYSIS BUTTON
    resultHTML += `
        <div style="margin-top: 20px; text-align: center;">
            <button onclick="newAnalysis()" 
                    style="padding: 12px 30px;display:none; background: #9b59b6; color: white;
                           border: none; border-radius: 25px; cursor: pointer;
                           font-size: 16px; font-weight: bold;">
                Analyze Another Image
            </button>
        </div>
    `;
    
    resultHTML += `</div>`;
    
    results_text.html(resultHTML);
    arrow.css("animation", "");
    
    // Store result in localStorage for details page
    if (data.detailed_results) {
        localStorage.setItem('detectionResult', JSON.stringify({
            image_name: data.image_name,
            final_decision: data.final_decision,
            final_confidence: finalConfidence,
            fake_probability: fakeProbability,
            real_probability: realProbability,
            detailed: data.detailed_results,
            heatmap: data.heatmap
        }));
    }
    
    if (window.innerWidth < 701) {
        results_text.off('click').on('click', goBack);
    }
}

function viewDetailedAnalysis() {
    // Open detailed analysis page
    window.open('/details', '_blank');
}

function newAnalysis() {
    resetUI();
}

function resetUI() {
    input.val('');
    imagePreview.src = '';
    imagePreview.style.display = 'none';
    results_text.html("Awaiting image upload...");
    currentResult = null;
    
    if (window.innerWidth < 701) {
        results_text.css('cursor', 'pointer');
        results_text.off('click').on('click', goBack);
    }
    
    arrow.css("animation", "");
}

// Keep existing functions
function changeType() {
    imagePreview.src = '';
    imagePreview.style.display = 'none';
    text.css("animation", "");
    input.css("animation", "");
    input.addClass("file-upload");
    
    setTimeout(() => {
        input.css("animation", "swap1 0.2s 1 linear");
        input.addClass("file-upload");
        text.css("animation", "swap2 0.2s 1 linear");
    }, 2);
    
    arrow.css("animation", "");
    
    setTimeout(() => {
        if (input.prop('type') == "file") {
            input.attr('id', "imageInput");
            input.prop('type', 'url');
            input.attr('placeholder', 'Paste image URL here');
            text.html("&nbsp;&nbsp;&nbsp;Upload Image&nbsp&nbsp;&nbsp;");
            input.css("margin-left", '0px');
            input.removeClass("file-upload");
            input.addClass("url-fix");
        } else {
            input.attr('id', "");
            text.css("animation", "");
            input.prop('type', 'file');
            input.attr('placeholder', '');
            text.html('Use URL Instead');
            input.css("margin-left", '40px');
            input.removeClass("url-fix");
            input.addClass("file-upload");
        }
    }, 200);
}

// Add spinner CSS
$(document).ready(function() {
    $('<style>').text(`
        .spinner {
            border: 5px solid rgba(52, 152, 219, 0.2);
            border-radius: 50%;
            border-top: 5px solid #3498db;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `).appendTo('head');
});
// Add URL validation function
function isValidUrl(url) {
    try {
        // Basic URL validation
        const urlObj = new URL(url);
        return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
    } catch (e) {
        return false;
    }
}

// Update analyzeImage function to validate URLs
function analyzeImage() {
    if (isAnalyzing) return;
    
    const formData = new FormData();
    
    if (input.prop('type') === 'file' && input[0].files.length > 0) {
        // File upload
        const file = input[0].files[0];
        if (file.size > 16 * 1024 * 1024) {
            alert("File is too large! Maximum size is 16MB.");
            return;
        }
        formData.append('image', file);
    } else if (input.prop('type') === 'url' && input.val().trim() !== '') {
        // URL input with validation
        const url = input.val().trim();
        
        // Basic validation
        if (!isValidUrl(url)) {
            alert("Please enter a valid URL (e.g., https://example.com/image.jpg)\n\nFor best results, use direct image links ending with .jpg, .png, etc.");
            arrow.css("animation", "");
            return;
            
        }
        
        // Show URL being processed
        const shortUrl = url.length > 50 ? url.substring(0, 50) + '...' : url;
        alert(`Processing URL: ${shortUrl}\n\nNote: Some websites block direct image downloads.`);
        
        formData.append('image_url', url);
    } else {
        alert("Please provide an image first!");
        return;
    }
    
    isAnalyzing = true;
    showLoadingState();
    
    fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        isAnalyzing = false;
        
        if (data.success) {
            currentResult = data;
            displayResults(data);
        } else {
            showErrorState(data.error || "Analysis failed");
        }
    })
    .catch(error => {
        isAnalyzing = false;
        console.error('Error:', error);
        showErrorState("Connection error. Please check server.");
    });
}
//test
if(window.innerWidth > 700){
    light.css("position" , "absolute");
    light.css("background-color", "white");
    light.css("height" , "200%");
    light.css("width" , "25px");
    light.css("margin" , "-15px 0 0 -3px");
    light.css("animation" , "light 1.5s linear infinite");
    light.css("trasnform" , "rotate-x(15deg)");
    results_text.addClass("light-effect");
}
