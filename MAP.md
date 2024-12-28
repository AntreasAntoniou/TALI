# TALI Project Map

## 🎯 Core Functionality

### Data Processing (`tali/data.py`)
- Dataset loading and management
- Transform pipeline implementation
- Modality type definitions
- Configuration system

### Frame Processing (`tali/frames.py`)
- Video frame extraction
- Frame selection methods
- Format conversion utilities

### Utilities (`tali/utils.py`)
- Helper functions
- Logging setup
- Common utilities

## 📦 Package Structure

### Main Package
```
tali/
├── data.py              # Core data processing
├── frames.py            # Video frame handling
├── utils.py             # Utility functions
├── data/               # Data-specific modules
├── demo/               # Demo applications
└── __init__.py         # Package initialization
```

### Examples and Tests
```
TALI/
├── examples.py          # Usage examples
├── tests/              # Test suite
└── requirements.txt    # Dependencies
```

## 🔧 Configuration Files

### Development Setup
- `requirements.txt` - Core dependencies
- `requirements_dev.txt` - Development dependencies
- `setup.py` - Package setup
- `.cursorrules` - Development guidelines

### Documentation
- `README.md` - Project overview
- `ARCHITECTURE.md` - System design
- `MILESTONES.md` - Development progress
- `MAP.md` - Project navigation

### Docker Support
- `Dockerfile` - Container definition
- `install-via-conda.sh` - Conda setup

## 🚀 Quick Links

### Getting Started
1. [Installation Guide](README.md#installation)
2. [Usage Examples](examples.py)
3. [API Documentation](ARCHITECTURE.md#core-components)

### Development
1. [Project Structure](ARCHITECTURE.md#directory-structure)
2. [Contributing Guidelines](CODE_OF_CONDUCT.md)
3. [Development Setup](README.md#development)

### Features
1. [Data Loading](ARCHITECTURE.md#data-loading-system)
2. [Transform Pipeline](ARCHITECTURE.md#transform-pipeline)
3. [Modality Support](ARCHITECTURE.md#modality-types)

## 🎓 Learning Resources

### Documentation
- [Architecture Overview](ARCHITECTURE.md)
- [Development Milestones](MILESTONES.md)
- [Code Documentation](tali/) 