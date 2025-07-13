// Service worker for Chrome extension
console.log('Background service worker loaded');

// Function to trigger the pipeline
async function triggerPipeline() {
    console.log('Triggering pipeline...');
    
    try {
        console.log('Making fetch request to http://127.0.0.1:5000/run-pipeline');
        
        const response = await fetch('http://127.0.0.1:5000/run-pipeline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ trigger: true })
        });
        
        console.log('Fetch request sent, response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        
        if (response.ok) {
            const responseData = await response.json();
            console.log('Pipeline response received:', responseData);
            
            return { success: true, data: responseData };
        } else {
            const errorText = await response.text();
            console.error('Response error:', errorText);
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }
    } catch (error) {
        console.error('Error triggering pipeline:', error);
        console.error('Error details:', error.message);
        return { success: false, error: error.message };
    }
}

// Function to generate diagram with LLaMA
async function generateDiagram(description) {
    console.log('🎨 Generating diagram for description:', description);
    
    try {
        console.log('Making fetch request to http://127.0.0.1:5000/generate-diagram');
        
        const response = await fetch('http://127.0.0.1:5000/generate-diagram', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ 
                description: description,
                source: 'chrome-extension'
            })
        });
        
        console.log('🎨 Diagram fetch request sent, response status:', response.status);
        console.log('🎨 Response headers:', [...response.headers.entries()]);
        
        if (response.ok) {
            const responseData = await response.json();
            console.log('🎨 Diagram response received:', responseData);
            
            return { success: true, data: responseData };
        } else {
            const errorText = await response.text();
            console.error('🎨 Diagram response error:', errorText);
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }
    } catch (error) {
        console.error('🎨 Error generating diagram:', error);
        console.error('🎨 Error details:', error.message);
        return { success: false, error: error.message };
    }
}

// Handle messages from popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    console.log('Message received in background:', request);
    
    if (request.action === 'triggerPipeline') {
        triggerPipeline().then(result => {
            sendResponse(result);
        });
        return true; // Will respond asynchronously
    } else if (request.action === 'generateDiagram') {
        generateDiagram(request.description).then(result => {
            sendResponse(result);
        });
        return true; // Will respond asynchronously
    }
});

console.log('Background service worker setup complete'); 