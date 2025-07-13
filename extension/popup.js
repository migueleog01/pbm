document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('startListening');
    const buttonText = document.getElementById('buttonText');
    const statusDiv = document.getElementById('status');
    const debugPanel = document.getElementById('debugPanel');
    const debugContent = document.getElementById('debugContent');
    const toggleDebugButton = document.getElementById('toggleDebug');
    const testServerButton = document.getElementById('testServer');
    
    let recognition = null;
    let isListening = false;
    let debugVisible = false;
    
    // Function to update status display with animation
    function updateStatus(message, type = 'listening') {
        statusDiv.innerHTML = getStatusHTML(message, type);
        statusDiv.className = `status ${type} slide-in`;
        statusDiv.style.display = 'block';
    }
    
    // Function to get status HTML with appropriate icons
    function getStatusHTML(message, type) {
        let icon = '';
        let waves = '';
        
        switch(type) {
            case 'listening':
                icon = '🎧';
                waves = '<span class="listening-indicator"><span class="wave"></span><span class="wave"></span><span class="wave"></span></span>';
                break;
            case 'success':
                icon = '✅';
                break;
            case 'error':
                icon = '❌';
                break;
        }
        
        return `${waves}${icon} ${message}`;
    }
    
    // Function to hide status with animation
    function hideStatus() {
        statusDiv.style.opacity = '0';
        setTimeout(() => {
            statusDiv.style.display = 'none';
            statusDiv.style.opacity = '1';
        }, 300);
    }
    
    // Function to update button state
    function updateButtonState(state, text) {
        const states = {
            idle: {
                text: 'Start Listening',
                disabled: false,
                classes: '',
                icon: '🎙️'
            },
            listening: {
                text: 'Listening...',
                disabled: true,
                classes: 'pulse-animation',
                icon: '🎧'
            },
            processing: {
                text: 'Processing...',
                disabled: true,
                classes: 'pulse-animation',
                icon: '⚡'
            }
        };
        
        const currentState = states[state];
        startButton.disabled = currentState.disabled;
        startButton.className = currentState.classes;
        buttonText.textContent = currentState.text;
        
        // Update icon
        const iconElement = startButton.querySelector('.icon-mic');
        iconElement.textContent = currentState.icon;
    }
    
    // Function to check microphone permissions
    async function checkMicrophonePermissions() {
        try {
            console.log('Checking microphone permissions...');
            
            // Check if permissions API is available
            if (navigator.permissions) {
                const permission = await navigator.permissions.query({name: 'microphone'});
                console.log('Microphone permission state:', permission.state);
                return permission.state;
            } else {
                console.log('Permissions API not available');
                return 'unavailable';
            }
        } catch (error) {
            console.error('Error checking microphone permissions:', error);
            return 'error';
        }
    }
    
    // Function to request microphone access - Chrome extension approach
    async function requestMicrophoneAccess() {
        try {
            console.log('Checking Chrome extension microphone access...');
            
            // For Chrome extensions, we rely on the webkitSpeechRecognition API
            // which handles permissions internally
            if ('webkitSpeechRecognition' in window) {
                console.log('webkitSpeechRecognition available - permissions should be handled automatically');
                return true;
            } else {
                console.log('webkitSpeechRecognition not available');
                return false;
            }
        } catch (error) {
            console.error('Error checking microphone access:', error);
            return false;
        }
    }

    // Function to initialize speech recognition
    function initializeSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            console.log('Initializing speech recognition...');
            recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            recognition.maxAlternatives = 1;
            
            // Add more logging for configuration
            console.log('Speech recognition configuration:');
            console.log('- continuous:', recognition.continuous);
            console.log('- interimResults:', recognition.interimResults);
            console.log('- lang:', recognition.lang);
            console.log('- maxAlternatives:', recognition.maxAlternatives);
            
            recognition.onstart = function() {
                console.log('Speech recognition started successfully');
                isListening = true;
                updateStatus('Listening for "hey computer"... (speak now)', 'listening');
                updateButtonState('listening');
            };
            
            recognition.onresult = function(event) {
                console.log('🎯 ONRESULT CALLED - Speech recognition result event:', event);
                console.log('Event results length:', event.results ? event.results.length : 'undefined');
                
                try {
                    if (!event.results || event.results.length === 0) {
                        console.log('❌ No results in speech recognition event');
                        updateStatus('Speech detected but no text recognized', 'error');
                        return;
                    }
                    
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const result = event.results[i];
                        
                        console.log(`Processing result ${i}:`, result);
                        
                        if (!result) {
                            console.log('❌ Null result at index', i);
                            continue;
                        }
                        
                        if (!result[0] || !result[0].transcript) {
                            console.log('❌ No transcript in result[0] at index', i);
                            continue;
                        }
                        
                        const transcript = result[0].transcript.toLowerCase().trim();
                        const originalTranscript = result[0].transcript.trim(); // Keep original case for display
                        const isFinal = result.isFinal || false;
                        
                        // Show EVERYTHING we're hearing in real-time
                        console.log(`🎙️ ${isFinal ? 'FINAL' : 'INTERIM'} SPEECH:`, `"${originalTranscript}"`);
                        updateStatus(`Hearing: "${originalTranscript}" ${isFinal ? '✓' : '...'}`, 'listening');
                        
                        // Check for hotword in both interim and final results
                        if (transcript.includes('hey computer') || transcript.includes('hey') || transcript.includes('computer')) {
                            console.log('🎯 HOTWORD DETECTED in transcript:', transcript);
                            
                            // Check if this same sentence contains the description after "hey computer"
                            if (transcript.includes('hey computer') && transcript.length > 'hey computer'.length + 5) {
                                // Extract the description part after "hey computer"
                                const description = transcript.replace(/^.*hey computer\s*/, '').trim();
                                
                                if (description && description.length > 3 && isFinal) {
                                    console.log('🚀 HOTWORD + DESCRIPTION IN SAME SENTENCE:', description);
                                    
                                    // Stop listening
                                    if (recognition && isListening) {
                                        recognition.stop();
                                        isListening = false;
                                    }
                                    
                                    // Update UI
                                    updateStatus('Processing with LLaMA...', 'success');
                                    updateButtonState('processing');
                                    
                                    // Send diagram description to LLaMA
                                    triggerDiagramGeneration(description);
                                    break;
                                }
                            }
                            
                            // If no description in same sentence, wait for next one
                            if (!window.hotwordDetected) {
                                updateStatus('Hotword detected! Now describe your diagram...', 'success');
                                updateButtonState('listening');
                                window.hotwordDetected = true;
                                console.log('🎤 Ready for diagram description...');
                            }
                            break;
                        }
                        
                        // If hotword was detected in previous speech, process this speech as description
                        if (window.hotwordDetected && isFinal && transcript.length > 3) {
                            console.log('🚀 SENDING TO LLAMA (SEPARATE SENTENCE):', originalTranscript);
                            
                            // Stop listening
                            if (recognition && isListening) {
                                recognition.stop();
                                isListening = false;
                            }
                            
                            // Update UI
                            updateStatus('Processing with LLaMA...', 'success');
                            updateButtonState('processing');
                            
                            // Send diagram description to LLaMA
                            triggerDiagramGeneration(originalTranscript);
                            window.hotwordDetected = false;
                            break;
                        }
                    }
                } catch (error) {
                    console.error('Error processing speech recognition result:', error);
                    updateStatus('Error processing speech input', 'error');
                }
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                console.error('Full error event:', event);
                isListening = false;
                
                // Detailed error handling with specific error codes
                let errorMessage = '';
                let troubleshootingTip = '';
                
                switch(event.error) {
                                    case 'not-allowed':
                    errorMessage = 'ERROR_MIC_PERMISSION: Microphone access denied';
                    troubleshootingTip = 'Click the microphone icon in Chrome address bar and select "Always allow"';
                    break;
                    case 'no-speech':
                        errorMessage = 'ERROR_NO_SPEECH: No speech detected';
                        troubleshootingTip = 'Try speaking louder or check microphone';
                        break;
                    case 'audio-capture':
                        errorMessage = 'ERROR_AUDIO_CAPTURE: Audio capture failed';
                        troubleshootingTip = 'Check if microphone is working in other apps';
                        break;
                    case 'network':
                        errorMessage = 'ERROR_NETWORK: Network connection failed';
                        troubleshootingTip = 'Check internet connection';
                        break;
                    case 'service-not-allowed':
                        errorMessage = 'ERROR_SERVICE_BLOCKED: Speech service blocked';
                        troubleshootingTip = 'Speech recognition may be disabled in browser settings';
                        break;
                    case 'bad-grammar':
                        errorMessage = 'ERROR_BAD_GRAMMAR: Grammar configuration error';
                        troubleshootingTip = 'Extension configuration issue';
                        break;
                    case 'language-not-supported':
                        errorMessage = 'ERROR_LANGUAGE: Language not supported';
                        troubleshootingTip = 'Try changing language settings';
                        break;
                    default:
                        errorMessage = `ERROR_UNKNOWN: ${event.error}`;
                        troubleshootingTip = 'Unknown error occurred';
                }
                
                console.error('Error code:', errorMessage);
                console.error('Troubleshooting:', troubleshootingTip);
                
                updateStatus(`${errorMessage}<br><small>${troubleshootingTip}</small>`, 'error');
                
                // Show detailed debugging info
                console.log('Browser info:', navigator.userAgent);
                console.log('Microphone permissions:', navigator.permissions);
                console.log('Speech recognition support:', 'webkitSpeechRecognition' in window);
                
                setTimeout(() => {
                    hideStatus();
                    updateButtonState('idle');
                }, 8000); // Longer timeout for error messages
            };
            
            recognition.onend = function() {
                console.log('Speech recognition ended');
                isListening = false;
                if (startButton.disabled && !startButton.classList.contains('pulse-animation')) {
                    updateButtonState('idle');
                    updateStatus('Speech recognition stopped. Click "Start Listening" to try again.', 'listening');
                }
            };
            
            // Add more debugging events
            recognition.onaudiostart = function() {
                console.log('Audio capture started');
                updateStatus('Audio detected - listening for "hey computer"...', 'listening');
            };
            
            recognition.onaudioend = function() {
                console.log('Audio capture ended');
            };
            
            recognition.onsoundstart = function() {
                console.log('Sound detected');
                updateStatus('Sound detected - processing...', 'listening');
            };
            
            recognition.onsoundend = function() {
                console.log('Sound ended');
            };
            
            recognition.onspeechstart = function() {
                console.log('Speech detected');
                updateStatus('Speech detected - analyzing...', 'listening');
            };
            
            recognition.onspeechend = function() {
                console.log('Speech ended');
            };
            
            recognition.onnomatch = function(event) {
                console.log('🔍 ONNOMATCH CALLED - No speech match found:', event);
                updateStatus('Speech heard but not recognized - try speaking more clearly', 'error');
            };
            
            return true;
        } else {
            console.error('webkitSpeechRecognition not supported');
            return false;
        }
    }
    
    // Function to trigger pipeline
    function triggerPipeline() {
        console.log('Triggering pipeline...');
        
        chrome.runtime.sendMessage({
            action: 'triggerPipeline'
        }, function(result) {
            console.log('Pipeline result:', result);
            
            if (result && result.success) {
                updateStatus('Pipeline triggered successfully!', 'success');
                console.log('Pipeline response received:', result.data);
                
                // Auto-hide status after 3 seconds and reset button
                setTimeout(() => {
                    hideStatus();
                    updateButtonState('idle');
                }, 3000);
            } else {
                updateStatus(`Pipeline error: ${result ? result.error : 'Unknown error'}`, 'error');
                
                // Auto-hide status after 5 seconds and reset button
                setTimeout(() => {
                    hideStatus();
                    updateButtonState('idle');
                }, 5000);
            }
        });
    }
    
    // Function to send diagram description to LLaMA
    function triggerDiagramGeneration(description) {
        console.log('🎨 Generating diagram for:', description);
        
        chrome.runtime.sendMessage({
            action: 'generateDiagram',
            description: description
        }, function(result) {
            console.log('🎨 Diagram generation result:', result);
            
            if (result && result.success) {
                console.log('✅ Diagram generated successfully!');
                console.log('📊 Mermaid code:', result.data.mermaid);
                
                updateStatus(`✅ Diagram generated! Check console for Mermaid code.`, 'success');
                
                // Auto-hide status after 5 seconds and reset button
                setTimeout(() => {
                    hideStatus();
                    updateButtonState('idle');
                }, 5000);
            } else {
                updateStatus(`❌ Diagram generation error: ${result ? result.error : 'Unknown error'}`, 'error');
                
                // Auto-hide status after 7 seconds and reset button
                setTimeout(() => {
                    hideStatus();
                    updateButtonState('idle');
                }, 7000);
            }
        });
    }
    
    // Handle start listening button click
    startButton.addEventListener('click', async function() {
        console.log('Start Listening button clicked');
        console.log('Current URL:', window.location.href);
        console.log('Extension context:', chrome.runtime ? 'Available' : 'Not available');
        
        // Check if speech recognition is supported
        if (!('webkitSpeechRecognition' in window)) {
            updateStatus('ERROR_NO_WEBKIT: Speech recognition not supported in this browser', 'error');
            return;
        }
        
        console.log('Speech recognition is supported');
        
        // For Chrome extensions, skip getUserMedia and go directly to speech recognition
        updateStatus('Initializing speech recognition...', 'listening');
        console.log('Skipping getUserMedia for Chrome extension - webkitSpeechRecognition will handle permissions');
        
        // Initialize speech recognition if not already done
        if (!recognition) {
            console.log('Initializing speech recognition...');
            if (!initializeSpeechRecognition()) {
                updateStatus('ERROR_INIT: Failed to initialize speech recognition', 'error');
                return;
            }
        }
        
        // Update button to listening state
        updateButtonState('listening');
        updateStatus('Starting speech recognition...', 'listening');
        
        // Start listening with additional debugging
        try {
            console.log('Attempting to start speech recognition...');
            console.log('Recognition object:', recognition);
            console.log('Recognition state before start:', recognition.state || 'unknown');
            
            recognition.start();
            console.log('Speech recognition start() called successfully');
            
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            console.error('Error type:', error.name);
            console.error('Error message:', error.message);
            updateStatus(`ERROR_START: ${error.name} - ${error.message}`, 'error');
            updateButtonState('idle');
        }
        
        // TODO: For testing server connection without speech recognition, uncomment this:
        /*
        // For now, skip speech recognition and directly test server connection
        updateButtonState('processing');
        updateStatus('Testing server connection...', 'listening');
        
        // Directly trigger pipeline to test connection
        console.log('Directly triggering pipeline for testing...');
        triggerPipeline();
        */
    });
    
    // Debug panel functionality
    function updateDebugInfo() {
        const debugInfo = {
            'Browser': navigator.userAgent,
            'Speech Recognition': 'webkitSpeechRecognition' in window ? 'Supported' : 'Not supported',
            'Extension Context': chrome.runtime ? 'Available' : 'Not available',
            'Current URL': window.location.href,
            'Recognition State': recognition ? (recognition.state || 'initialized') : 'not initialized',
            'Is Listening': isListening,
            'MediaDevices': navigator.mediaDevices ? 'Available' : 'Not available',
            'Permissions API': navigator.permissions ? 'Available' : 'Not available'
        };
        
        let debugText = '';
        for (const [key, value] of Object.entries(debugInfo)) {
            debugText += `${key}: ${value}\n`;
        }
        
        debugContent.textContent = debugText;
    }
    
    function toggleDebugPanel() {
        debugVisible = !debugVisible;
        if (debugVisible) {
            debugPanel.style.display = 'block';
            toggleDebugButton.textContent = 'Hide Debug Info';
            updateDebugInfo();
        } else {
            debugPanel.style.display = 'none';
            toggleDebugButton.textContent = 'Show Debug Info';
        }
    }
    
    // Add debug toggle button event listener
    toggleDebugButton.addEventListener('click', toggleDebugPanel);
    
    // Test server connection button
    testServerButton.addEventListener('click', function() {
        console.log('Testing server connection...');
        updateStatus('Testing server connection...', 'listening');
        
        // Directly trigger pipeline to test connection (bypass speech recognition)
        triggerPipeline();
    });
    
    // Show debug panel by default since we're troubleshooting
    debugPanel.style.display = 'block';
    toggleDebugButton.textContent = 'Hide Debug Info';
    debugVisible = true;
    updateDebugInfo();
    
    // Update debug info every 2 seconds
    setInterval(updateDebugInfo, 2000);
    
    // Add some nice entrance animations
    setTimeout(() => {
        document.querySelector('.header').style.animation = 'slideIn 0.6s ease-out';
        document.querySelector('.main-section').style.animation = 'slideIn 0.6s ease-out 0.2s both';
    }, 100);
}); 