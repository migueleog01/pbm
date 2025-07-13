# 🎤 Voice Assistant Chrome Extension

A modern Chrome extension that listens for voice commands and triggers automated workflows through a local API server. Features a beautiful glassmorphism UI and seamless speech recognition integration.

## ✨ **Features**

- 🎙️ **Speech Recognition**: Uses webkitSpeechRecognition API to listen for voice commands
- 🔥 **Hotword Detection**: Responds to "hey computer" phrase
- 🌐 **API Integration**: Makes POST requests to local server when triggered
- 🎨 **Modern UI**: Beautiful glassmorphism design with smooth animations
- 🔄 **Real-time Feedback**: Live status updates and visual feedback
- 🛡️ **Secure**: Runs locally with no external data transmission

## 📁 **Project Structure**

```
/extension
├── manifest.json          # Chrome extension manifest (v3)
├── popup.html             # Modern popup interface  
├── popup.js               # Frontend logic and speech recognition
├── background.js          # Service worker for API calls
├── server.py              # Flask server for handling requests
├── requirements.txt       # Python dependencies
├── voice-assistant-env/   # Python virtual environment
├── icons/
│   └── icon.png          # Extension icon
└── README.md             # This file
```

## 🚀 **Quick Start**

### **1. Prerequisites**
- **Chrome Browser** (required for webkitSpeechRecognition)
- **Python 3.7+** installed on your system
- **Microphone access** for speech recognition

### **2. Download & Setup**
```bash
# Navigate to your project directory
cd /path/to/extension

# Create virtual environment
python3 -m venv voice-assistant-env

# Activate virtual environment
source voice-assistant-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Start the Server**
```bash
# Make sure virtual environment is activated
source voice-assistant-env/bin/activate

# Start the Flask server
python server.py
```

You should see:
```
🚀 Starting Voice Assistant Server...
📡 Server will run on: http://localhost:5000
🎤 Ready to receive voice commands!
```

### **4. Load Chrome Extension**
1. Open Chrome and go to `chrome://extensions/`
2. Enable **"Developer Mode"** (toggle in top right)
3. Click **"Load unpacked"**
4. Select the extension folder
5. **Pin the extension** to your toolbar (puzzle piece icon → pin "Voice Assistant")

### **5. Test the Extension**
1. **Click the Voice Assistant icon** in your Chrome toolbar
2. **Click "Test Connection"** button
3. **Check your server terminal** - you should see:
   ```
   🎤 [timestamp] BUTTON CLICKED - TESTING CONNECTION!
   📝 Request data: {"trigger": true}
   ✅ Pipeline triggered successfully!
   ```

## 🎯 **Current Status: Testing Mode**

The extension is currently in **testing mode** for easy setup verification:

- ✅ **Button directly triggers server** (no speech recognition)
- ✅ **Tests end-to-end connectivity**
- ✅ **Verifies API communication**
- ✅ **Shows beautiful UI animations**

### **To Enable Speech Recognition:**

In `popup.js`, uncomment the speech recognition code section and comment out the direct trigger. Look for:

```javascript
// TODO: Uncomment this section when ready to test speech recognition
/*
// Check if speech recognition is supported...
*/
```

## 🛠️ **Technical Details**

### **API Endpoint**
- **URL**: `http://127.0.0.1:5000/run-pipeline`
- **Method**: POST
- **Headers**: `Content-Type: application/json`
- **Body**: `{"trigger": true}`
- **Response**: `{"status": "success", "message": "Pipeline executed successfully", ...}`

### **Chrome Extension Architecture**
- **Manifest V3**: Modern Chrome extension format
- **Service Worker**: Handles background API calls
- **Popup Script**: Manages UI and speech recognition
- **CORS Enabled**: Proper cross-origin resource sharing
- **Host Permissions**: Access to local server endpoints

### **Security Features**
- 🔒 **Local Only**: All communication stays on your machine
- 🔐 **No External Requests**: No data sent to external servers
- 🛡️ **Sandbox Mode**: Extension runs in Chrome's security sandbox
- 🎯 **Specific Permissions**: Only requests necessary permissions

## 🔧 **Configuration Options**

### **Change Server Port**
In `server.py`, modify:
```python
app.run(port=5000)  # Change to desired port
```

In `background.js`, update:
```javascript
fetch('http://127.0.0.1:NEW_PORT/run-pipeline', {
```

### **Customize Hotword**
In `popup.js`, change the detection phrase:
```javascript
if (transcript.includes('your custom phrase')) {
```

### **Modify UI Text**
In `popup.html`, update titles and descriptions:
```html
<h1>Your Custom Title</h1>
<p class="subtitle">Your custom subtitle</p>
```

## 🐛 **Troubleshooting**

### **403 Permission Errors**
```bash
# Solution: Reload the extension after any manifest.json changes
# Go to chrome://extensions/ → click reload button
```

### **Port 5000 Already in Use**
```bash
# Kill existing processes
pkill -f "python server.py"

# Or use different port
python server.py --port 5001
```

### **Speech Recognition Not Working**
- Ensure you're using **Chrome browser** (not Firefox/Safari)
- **Allow microphone permissions** when prompted
- Check that microphone is working in other applications
- Try speaking clearly and closer to microphone

### **Extension Not Loading**
- Verify all files are in the same directory
- Check Chrome Developer Tools console for errors
- Ensure `manifest.json` is valid JSON
- Try reloading the extension

### **Server Connection Failed**
```bash
# Test server manually
curl -X POST -H "Content-Type: application/json" -d '{"trigger": true}' http://127.0.0.1:5000/run-pipeline

# Should return: {"status": "success", ...}
```

## 🔍 **Debugging**

### **Chrome Extension Console**
1. Right-click extension icon → "Inspect popup"
2. Check Console tab for JavaScript errors
3. Look for fetch request logs and responses

### **Server Logs**
Monitor the terminal running `server.py` for:
- ✅ Successful requests with detailed headers
- ❌ Error messages with stack traces
- 📊 Request count and timing

### **Network Inspection**
1. Open Chrome DevTools (F12)
2. Go to Network tab
3. Click extension button
4. Look for POST request to `127.0.0.1:5000`

## 📊 **Usage Examples**

### **Basic API Response**
```json
{
  "status": "success",
  "message": "Pipeline executed successfully",
  "timestamp": "2025-01-19 12:00:00",
  "run_count": 1
}
```

### **Error Response**
```json
{
  "status": "error",
  "message": "Invalid trigger data"
}
```

## 🚀 **Next Steps / Customization**

1. **Enable Speech Recognition**: Uncomment the speech code in `popup.js`
2. **Custom Commands**: Add more voice command detection
3. **API Integration**: Connect to your specific workflow/automation system
4. **UI Customization**: Modify colors, animations, and layout
5. **Additional Features**: Add settings, multiple hotwords, etc.

## 📝 **Dependencies**

### **Python Requirements** (`requirements.txt`)
```
flask>=2.0.0
flask-cors>=3.0.0
```

### **Browser Requirements**
- **Chrome 70+** (for webkitSpeechRecognition)
- **Manifest V3 support**
- **Microphone permissions**

## 💡 **Tips for Success**

1. **Always reload the extension** after making changes to manifest.json
2. **Use 127.0.0.1 instead of localhost** for better Chrome compatibility
3. **Check both extension console AND server terminal** when debugging
4. **Test server connectivity first** before enabling speech recognition
5. **Keep server running** while using the extension

## 🎉 **Current State: WORKING**

✅ **Server communication**: Fully functional  
✅ **Chrome extension**: Loaded and operational  
✅ **API calls**: Successfully sending/receiving  
✅ **UI animations**: Beautiful and responsive  
✅ **Error handling**: Comprehensive error messages  

The extension is ready for speech recognition integration or further customization!

---

*Created with ❤️ for seamless voice-controlled automation* 