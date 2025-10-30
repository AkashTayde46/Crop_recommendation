# Smart Health Diagnostics - Crop Recommendation System

This is a machine learning-powered crop recommendation system that helps farmers choose the best crops based on soil conditions, weather data, and environmental factors.

## Features

- **AI-Powered Recommendations**: Get personalized crop recommendations using machine learning
- **Comprehensive Data Input**: Soil NPK values, weather conditions, pH levels, and rainfall data
- **Real-time Analysis**: Instant predictions based on your input data
- **Modern Web Interface**: Beautiful, responsive React frontend
- **RESTful API**: JSON API for integration with other applications

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher (for React frontend)
- The `crop_recommendation.pkl` model file

### Backend Setup (Flask API)

1. **Navigate to the Crop_Recommendation directory:**
   ```bash
   cd Crop_Recommendation
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server:**
   ```bash
   python app.py
   ```
   
   Or use the setup script:
   ```bash
   python run_server.py
   ```

4. **Verify the server is running:**
   - API endpoint: http://localhost:5000/predict
   - Web interface: http://localhost:5000/

### Frontend Setup (React)

1. **Navigate to the client directory:**
   ```bash
   cd client
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```

4. **Access the application:**
   - Main app: http://localhost:3000
   - Diagnostics: http://localhost:3000/diagnostics

## API Usage

### Endpoint: POST /predict

**Request Body (JSON):**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.9,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202.9,
  "ph_category": "Neutral",
  "rainfall_level": "High"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "rice",
  "confidence": "High"
}
```

### Input Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| N | number | Nitrogen content in soil | 90 |
| P | number | Phosphorus content in soil | 42 |
| K | number | Potassium content in soil | 43 |
| temperature | number | Temperature in Celsius | 20.9 |
| humidity | number | Humidity percentage | 82 |
| ph | number | Soil pH level | 6.5 |
| rainfall | number | Rainfall in mm | 202.9 |
| ph_category | string | pH category (Neutral/Alkaline/Acidic) | "Neutral" |
| rainfall_level | string | Rainfall level (Very High/High/Medium/Low) | "High" |

### Supported Crops

The model can recommend 22 different crops:
- rice, maize, chickpea, kidneybeans, pigeonpeas
- mothbeans, mungbean, blackgram, lentil, pomegranate
- banana, mango, grapes, watermelon, muskmelon
- apple, orange, papaya, coconut, cotton
- jute, coffee

## Model Information

- **Algorithm**: Machine Learning model trained on agricultural data
- **Features**: 7 primary features + 4 derived features
- **Model File**: `crop_recommendation.pkl`
- **Framework**: scikit-learn

## Troubleshooting

### Common Issues

1. **Model file not found:**
   - Ensure `crop_recommendation.pkl` is in the Crop_Recommendation directory
   - Check file permissions

2. **CORS errors:**
   - The Flask app includes CORS headers for cross-origin requests
   - Ensure both frontend and backend are running

3. **Port conflicts:**
   - Flask runs on port 5000 by default
   - React runs on port 3000 by default
   - Change ports in the respective configuration files if needed

### Error Handling

The API returns appropriate error messages for:
- Invalid input data
- Missing required fields
- Server errors
- Model prediction failures

## Development

### Adding New Features

1. **Backend**: Modify `app.py` to add new endpoints or features
2. **Frontend**: Update React components in `client/src/pages/`
3. **API**: Extend the API service in `client/src/services/`

### Testing

Test the API using curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82,"ph":6.5,"rainfall":202.9,"ph_category":"Neutral","rainfall_level":"High"}'
```

## License

This project is part of the AgriAI application suite.
