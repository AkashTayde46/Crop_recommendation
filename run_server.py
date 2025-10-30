#!/usr/bin/env python3
"""
Setup and run the Smart Health Diagnostics Flask server
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_model_file():
    """Check if the model file exists"""
    model_path = "crop_recommendation.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the crop_recommendation.pkl file is in the Crop_Recommendation directory")
        return False
    print("✅ Model file found!")
    return True

def run_server():
    """Run the Flask server"""
    print("Starting Smart Health Diagnostics server...")
    print("🌱 Server will be available at: http://localhost:5000")
    print("📊 API endpoint: http://localhost:5000/predict")
    print("🌐 Web interface: http://localhost:5000/")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
    except Exception as e:
        print(f"❌ Error running server: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Smart Health Diagnostics Server")
    print("=" * 50)
    
    if not check_model_file():
        sys.exit(1)
    
    if not install_requirements():
        sys.exit(1)
    
    print("\n🎯 Starting server...")
    run_server()
