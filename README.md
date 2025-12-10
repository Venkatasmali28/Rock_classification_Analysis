# MineSafety - Mine Safety Analysis Platform

A comprehensive machine learning-based platform for mine safety analysis, combining rock classification, crack detection, and equipment anomaly prediction. Built with Flask backend and React frontend.

## Project Overview

MineSafety is an intelligent system designed to enhance safety in mining operations through:
- **Rock Classification**: Automatically classify rock types from images (7 categories across 3 geological categories)
- **Crack Segmentation**: Detect and segment cracks in mining structures
- **Equipment Anomaly Detection**: Predict equipment faults and maintenance needs
- **Machine Life Prediction**: Estimate remaining useful life of mining equipment

## Features

### Rock Classification
- Classifies rocks into 7 categories:
  - Igneous: Basalt, Granite
  - Metamorphic: Marble, Quartzite
  - Sedimentary: Coal, Limestone, Sandstone
- Uses ResNet18 deep learning model
- Confidence threshold-based predictions (>60% confidence required)
- Visual explanations using Grad-CAM

### Crack Segmentation
- Detects and segments cracks in mining structures
- U-Net based semantic segmentation
- Real-time crack analysis
- Mask visualization output

### Equipment Anomaly Detection
- Machine learning-based fault prediction
- Random Forest classifier for equipment failures
- Location-based fault analysis
- Predictive maintenance recommendations

## Project Structure

```
MineSafety/
├── backend/                          # Flask API server
│   ├── app.py                        # Main Flask application
│   ├── requirements.txt              # Python dependencies
│   ├── split_classification_dataset.py
│   ├── split_segmentation_dataset.py
│   ├── data/                         # Dataset directories
│   │   ├── rock_classification/     # Original rock classification data
│   │   ├── rock_classification_split/ # Split train/val data
│   │   ├── crack_segmentation/      # Original crack segmentation data
│   │   └── crack_segmentation_split/ # Split train/val data
│   ├── models/                       # ML models and training scripts
│   │   ├── rock_classifier.py       # Rock classification model
│   │   ├── crack_segmenter.py       # Crack segmentation model
│   │   ├── train_classifier.py      # Rock classifier training
│   │   ├── train_crack_segmentation.py
│   │   ├── train_machinelife.py     # Equipment life prediction training
│   │   ├── unet.py                  # U-Net architecture
│   │   ├── equipment_anomaly_data_india_balanced.csv
│   │   └── saved_models/            # Pre-trained model weights
│   │       ├── rock_classifier.pth
│   │       └── crack_segmenter.pth
│   └── utils/                        # Utility functions
│       ├── preprocessing.py          # Image preprocessing
│       └── gradcam.py               # Grad-CAM visualization
│
└── frontend/                         # React + Vite application
    ├── src/
    │   ├── App.jsx                  # Main application component
    │   ├── main.jsx                 # Entry point
    │   ├── pages/
    │   │   ├── Home.jsx            # Home page
    │   │   ├── RockAnalyser.jsx    # Rock classification interface
    │   │   └── MachineLife.jsx     # Equipment prediction interface
    │   ├── components/
    │   │   └── Footer.jsx          # Footer component
    │   └── assets/
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Tech Stack

### Backend
- **Framework**: Flask
- **Deep Learning**: PyTorch, TorchVision
- **Image Processing**: PIL, OpenCV
- **Machine Learning**: Scikit-learn
- **Database**: Python pickle (model serialization)

### Frontend
- **Framework**: React 19.1.1
- **Build Tool**: Vite
- **Styling**: Tailwind CSS 4.1.17
- **Animation**: Framer Motion 12.23.24
- **Routing**: React Router DOM 7.9.5

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure pre-trained models are in `models/saved_models/`:
   - `rock_classifier.pth`
   - `crack_segmenter.pth`

4. Run the Flask server:
```bash
python app.py
```

The backend will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

### Rock Classification
- **POST** `/predict`
  - Uploads an image for rock classification
  - Returns: rock type, confidence score, and Grad-CAM visualization

### Crack Segmentation
- **POST** `/segment`
  - Uploads an image for crack detection
  - Returns: segmentation mask and confidence metrics

### Equipment Anomaly
- **POST** `/equipment-fault`
  - Predicts equipment faults
  - Returns: fault type, location, and severity

### Health Check
- **GET** `/`
  - Returns server status

## Dataset Information

### Rock Classification Dataset
- 7 rock type categories across 3 geological classes
- Organized by type and subtype in `data/rock_classification/`
- Split into train/validation sets in `data/rock_classification_split/`

### Crack Segmentation Dataset
- Images with corresponding mask annotations
- Located in `data/crack_segmentation/`
- Train/test/validation splits available
- Masks in `masks/` directories

## Model Details

### Rock Classifier
- **Architecture**: ResNet18
- **Output Classes**: 7 rock types
- **Confidence Threshold**: 60%
- **Input Size**: Standard ImageNet preprocessing
- **Weights**: `rock_classifier.pth`

### Crack Segmenter
- **Architecture**: U-Net
- **Input Size**: 224x224
- **Output**: Binary segmentation mask
- **Threshold**: 0.5 probability
- **Weights**: `crack_segmenter.pth`

### Equipment Fault Predictor
- **Algorithm**: Random Forest Classifier
- **Input Features**: Equipment specifications, location, usage patterns
- **Model File**: `model.pkl`

## Usage Examples

### Via Frontend
1. Open the application at `http://localhost:5173`
2. Navigate to Rock Analyser for rock classification
3. Navigate to Machine Life for equipment prediction
4. Upload images or input equipment data for predictions

### Via API
```bash
# Rock Classification
curl -X POST http://localhost:5000/predict \
  -F "image=@rock_image.jpg"

# Equipment Fault Prediction
curl -X POST http://localhost:5000/equipment-fault \
  -H "Content-Type: application/json" \
  -d '{"equipment":"pump","location":"entrance","hours_used":5000}'
```

## Configuration

### Backend
- CORS enabled for localhost:5173 and localhost:3000
- PyTorch device auto-detection (CUDA/CPU)
- Model weights paths defined in respective model files

### Frontend
- Vite configuration in `vite.config.js`
- ESLint configuration in `eslint.config.js`
- Tailwind CSS configuration via @tailwindcss/vite

## Data Preprocessing

Image preprocessing pipeline includes:
- Normalization to ImageNet statistics
- Resizing to model input dimensions
- Tensor conversion for PyTorch

See `utils/preprocessing.py` for implementation details.

## Visualization

### Grad-CAM
- Visual explanations for rock classification predictions
- Highlights important image regions
- Implementation in `utils/gradcam.py`

## Training Models

### Train Rock Classifier
```bash
cd backend
python models/train_classifier.py
```

### Train Crack Segmenter
```bash
cd backend
python models/train_crack_segmentation.py
```

### Train Equipment Predictor
```bash
cd backend
python models/train_machinelife.py
```

## Dataset Splitting

### Split Rock Classification Data
```bash
python split_classification_dataset.py
```

### Split Crack Segmentation Data
```bash
python split_segmentation_dataset.py
```

## Development

### Running Tests
```bash
cd frontend
npm run lint
```

### Building for Production

Frontend:
```bash
cd frontend
npm run build
```

## Performance Considerations

- Models run on GPU if available (auto-detected)
- Fallback to CPU for systems without CUDA
- Image resizing to fixed dimensions for efficiency
- Batch processing support for multiple predictions

## Security

- CORS configured for specific origins
- Input validation on file uploads
- Model weights protected in saved_models directory

## Troubleshooting

### Model Loading Issues
- Verify model files exist in `models/saved_models/`
- Check PyTorch device compatibility
- Ensure correct file paths in model Python files

### Image Upload Problems
- Verify image format (JPG, PNG supported)
- Check image size isn't exceeding limits
- Ensure proper CORS configuration

### Frontend Not Connecting
- Verify Flask backend is running on correct port
- Check CORS origin configuration in app.py
- Clear browser cache and refresh

## Future Enhancements

- Real-time video stream processing
- Multi-model ensemble predictions
- Advanced anomaly detection algorithms
- Mobile app integration
- Cloud deployment
- Database integration for historical data
- Advanced analytics dashboard

## License

This project is part of the MineSafety initiative for mine safety analysis.

## Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Repository**: Rock_classification_Analysis  
**Owner**: Venkatasmali28  
**Last Updated**: December 2025
