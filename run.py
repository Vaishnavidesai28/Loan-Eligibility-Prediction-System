"""
Simple script to run the Flask application
"""
from app import app
import webbrowser
import threading
import time

def open_browser():
    """Open the browser after a short delay to allow the server to start"""
    time.sleep(1.5)
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("=" * 50)
    print("Loan Eligibility Predictor")
    print("=" * 50)
    print("Server starting...")
    print("Opening browser at http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)

