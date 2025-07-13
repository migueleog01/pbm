# 🔧 Voice-to-Mermaid Chrome Extension Troubleshooting Guide

## 🚨 **Common Error Codes and Solutions**

### **ERROR_MIC_PERMISSION / not-allowed**
**Problem**: Chrome cannot access your microphone
**Solutions**:
1. Click the **microphone icon** in Chrome's address bar and allow access
2. Go to `chrome://settings/content/microphone` and ensure the site is allowed
3. Check if other apps (Zoom, Teams) are using your microphone
4. Restart Chrome after changing permissions
5. Try reloading the extension in `chrome://extensions/`

### **ERROR_NO_WEBKIT**
**Problem**: Speech recognition not supported in browser
**Solutions**:
1. Use **Chrome browser** (not Firefox, Safari, or Edge)
2. Update Chrome to the latest version
3. Check if you're using Chrome in incognito mode (may not work)

### **ERROR_MIC_DENIED**
**Problem**: Microphone access denied in browser settings
**Solutions**:
1. Go to `chrome://settings/privacy/microphone`
2. Make sure microphone access is enabled
3. Check if the extension is in the blocked list
4. Clear browser permissions and try again

### **ERROR_AUDIO_CAPTURE**
**Problem**: Audio capture failed
**Solutions**:
1. Test your microphone in other apps
2. Check macOS privacy settings: `System Preferences > Security & Privacy > Microphone`
3. Ensure Chrome has microphone permission in macOS
4. Try unplugging and reconnecting external microphones

### **ERROR_NETWORK**
**Problem**: Network connection failed
**Solutions**:
1. Check internet connection
2. Verify the server is running on `localhost:5000`
3. Try the "Test Server Connection" button
4. Check if firewall is blocking the connection

---

## 🔍 **Debugging Steps**

### **Step 1: Check Debug Panel**
The extension now shows a debug panel with:
- Browser information
- Speech recognition support
- Extension context
- Recognition state
- Permission status

### **Step 2: Test Server Connection**
1. Click "Test Server Connection" button
2. This bypasses speech recognition and tests the Flask server
3. Check the server terminal for connection logs

### **Step 3: Check Chrome DevTools**
1. Right-click the extension icon → "Inspect popup"
2. Check the Console tab for detailed error messages
3. Look for specific error codes and stack traces

### **Step 4: Verify Permissions**
1. Go to `chrome://extensions/`
2. Find "Voice to Mermaid" extension
3. Check if it has microphone permissions
4. Try reloading the extension

---

## 🛠️ **Advanced Troubleshooting**

### **Chrome Extension Manifest Issues**
If the extension won't load:
1. Check `manifest.json` for syntax errors
2. Ensure all required permissions are listed
3. Verify file paths are correct
4. Try reloading the extension

### **Speech Recognition Issues**
If recognition starts but fails:
1. Check if multiple speech recognition instances are running
2. Verify the hotword "hey computer" is being detected
3. Test with different microphones
4. Check microphone sensitivity settings

### **Server Connection Issues**
If the server isn't responding:
1. Verify Flask server is running on port 5000
2. Check for CORS errors in browser console
3. Test manually: `curl -X POST http://localhost:5000/run-pipeline`
4. Ensure the voice-to-mermaid script exists

---

## 📋 **Checklist for Common Issues**

### **Before Starting**
- [ ] Chrome browser (not other browsers)
- [ ] Chrome is up to date
- [ ] Microphone is working in other apps
- [ ] Flask server is running (`python server.py`)
- [ ] Extension is loaded and pinned to toolbar

### **Extension Issues**
- [ ] Extension has microphone permission
- [ ] Extension is enabled in `chrome://extensions/`
- [ ] No console errors in extension popup
- [ ] Debug panel shows "Speech Recognition: Supported"

### **Server Issues**
- [ ] Server running on `localhost:5000`
- [ ] "Test Server Connection" button works
- [ ] Voice-to-mermaid script exists in parent directory
- [ ] No CORS errors in browser console

### **Voice Recognition Issues**
- [ ] Microphone permission granted
- [ ] Speaking clearly and close to microphone
- [ ] Saying "hey computer" clearly
- [ ] No other apps using microphone simultaneously

---

## 🎯 **Quick Fix Commands**

### **Reset Chrome Permissions**
```bash
# Clear all Chrome permissions (will reset ALL sites)
rm -rf ~/Library/Application\ Support/Google/Chrome/Default/Preferences
```

### **Reset Extension**
1. Go to `chrome://extensions/`
2. Remove the extension
3. Reload the extension folder
4. Grant permissions again

### **Test Server Manually**
```bash
# Test if server is responding
curl -X POST -H "Content-Type: application/json" \
  -d '{"trigger": true}' \
  http://localhost:5000/run-pipeline
```

### **Check macOS Microphone Permissions**
```bash
# Check microphone permissions on macOS
tccutil check Microphone com.google.Chrome
```

---

## 📞 **Still Having Issues?**

If you're still experiencing problems:

1. **Check the debug panel** for specific error codes
2. **Open Chrome DevTools** and look for console errors
3. **Test the server connection** separately from speech recognition
4. **Try a different microphone** if available
5. **Use the error codes** above to identify the specific issue

The debug panel will show real-time information about:
- Browser compatibility
- Permission status
- Recognition state
- Server connectivity

---

## 🔄 **Reset Everything**

If nothing works, try this complete reset:

1. **Stop the server** (Ctrl+C)
2. **Remove the extension** from Chrome
3. **Clear Chrome permissions** for localhost
4. **Restart Chrome**
5. **Reload the extension**
6. **Grant permissions again**
7. **Start the server**
8. **Test step by step**

This should resolve most issues! 🎉 