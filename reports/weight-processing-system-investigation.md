# Investigation: Weight Measurement Processing System Architecture

## Bottom Line
**Root Cause**: Modular, stateful weight processing pipeline with Kalman filtering
**Core Purpose**: Process weight measurements with adaptive noise reduction and quality assessment
**Confidence**: High

## Executive Summary
This is a sophisticated weight measurement processing system designed for medical/health monitoring environments. It processes weight data from multiple sources with varying reliability, applies adaptive Kalman filtering for noise reduction, and includes intelligent outlier detection with quality-based override mechanisms. The system maintains persistent state per user and can adapt its behavior after gaps in measurements or source changes.

## System Architecture

### Data Flow Diagram
```
CSV Input → main.py → processor.py → [Validation] → [Quality Scoring] → [Kalman Filter] → [Database]
                           ↓                                                    ↓
                    [Outlier Detection] ←─────────────────→ [Reset Manager]
                           ↓
                    [Visualization] → HTML Dashboards
```

## Component-by-Component Breakdown

### 1. **main.py** (Entry Point)
- **Purpose**: Orchestrates CSV processing and visualization generation
- **Key Functions**:
  - `stream_process()`: Main pipeline orchestrator
  - `generate_single_visualization()`: Parallel visualization generation
- **Data Flow**: CSV → Results dictionary → Visualizations
- **Dependencies**: processor, database, visualization modules

### 2. **src/processing/processor.py** (Core Pipeline)
- **Purpose**: Central processing hub that coordinates all analysis steps
- **Key Functions**:
  - `process_measurement()`: Main processing pipeline (lines 86-502)
  - `check_and_reset_for_gap()`: Detects 30+ day gaps
  - `WeightProcessor` class: Legacy wrapper
- **Data Flow**: Raw weight → Preprocessing → State Management → Kalman → Quality → Result
- **Critical Logic**: Handles reset detection, adaptive parameter selection, state persistence

### 3. **src/processing/kalman.py** (Adaptive Filtering)
- **Purpose**: Implements adaptive Kalman filter for noise reduction
- **Key Components**:
  - `KalmanFilterManager`: Core filter operations
  - Three reset types: INITIAL, HARD (30+ day gaps), SOFT (manual entries)
  - Adaptive parameters that decay over time
- **Data Flow**: Weight + State → Filter → Prediction + Covariance
- **Key Feature**: Source-specific noise multipliers (0.5-3.0x)

### 4. **src/processing/quality_scorer.py** (Quality Assessment)
- **Purpose**: Multi-factor quality scoring system
- **Components**:
  - Safety score (35%): Physiological limits
  - Plausibility score (25%): BMI detection
  - Consistency score (25%): Change rate validation
  - Reliability score (15%): Source-based
- **Data Flow**: Measurement → Score (0-1) → Accept/Reject decision
- **Override**: High quality (>0.8) can override outlier detection

### 5. **src/processing/outlier_detection.py** (Statistical Analysis)
- **Purpose**: Batch outlier detection with quality override
- **Methods**:
  - IQR-based detection
  - Modified Z-score (MAD)
  - Temporal consistency checks
- **Key Feature**: 15% Kalman deviation threshold
- **Protection**: High-quality measurements immune to outlier rejection

### 6. **src/processing/validation.py** (Data Validation)
- **Purpose**: Multi-layer validation system
- **Classes**:
  - `PhysiologicalValidator`: Hard limits and rate checks
  - `BMIValidator`: BMI-based plausibility
  - `ThresholdCalculator`: Dynamic threshold computation
  - `DataQualityPreprocessor`: Initial data cleaning
- **Critical**: Enforces safety boundaries and physiological constraints

### 7. **src/processing/reset_manager.py** (State Reset Logic)
- **Purpose**: Manages Kalman filter resets with different strategies
- **Reset Types**:
  - INITIAL: First measurement (most aggressive)
  - HARD: After 30+ day gaps (aggressive)
  - SOFT: Manual data entry (gentle)
- **Key Logic**: Different adaptation parameters per reset type

### 8. **src/database/database.py** (State Persistence)
- **Purpose**: User state management and persistence
- **Key State Elements**:
  - Kalman state and covariance
  - Last measurements and timestamps
  - Reset events and adaptation parameters
  - 30-measurement history buffer
- **Operations**: CRUD operations with numpy serialization

### 9. **src/viz/visualization.py** (Data Visualization)
- **Purpose**: Generate interactive HTML dashboards
- **Features**:
  - Weight timeline with Kalman predictions
  - Quality score visualization
  - Innovation/residual plots
  - Source-based color coding
- **Technology**: Plotly-based interactive charts

### 10. **src/config_loader.py** (Configuration System)
- **Purpose**: Profile-based configuration management
- **Profiles**: strict, moderate, lenient, clinical, balanced
- **Mappings**:
  - Filtering strength
  - Adaptation speed
  - Trust levels
  - Gap sensitivity
- **Override**: Allows fine-tuning via config.toml

## Feature Catalog

### Core Processing
1. **Adaptive Kalman Filtering**: Noise reduction with source-specific parameters
2. **Multi-Source Support**: Handles 6+ data sources with reliability profiles
3. **Quality Scoring**: 4-component quality assessment system
4. **Outlier Detection**: Statistical analysis with quality override
5. **Reset Management**: Intelligent adaptation after gaps or source changes

### Data Quality
1. **Physiological Validation**: Hard limits (30-400kg)
2. **BMI Plausibility**: Detects impossible BMI values
3. **Rate Limiting**: Max daily change constraints
4. **Source Reliability**: Noise multipliers (0.5x-3.0x)
5. **Time-based Thresholds**: Hourly/daily/weekly limits

### State Management
1. **User State Persistence**: SQLite-based storage
2. **Measurement History**: 30-measurement buffer
3. **Reset Event Tracking**: Full audit trail
4. **Adaptive Parameters**: Time/count-based decay

### Visualization
1. **Interactive Dashboards**: HTML/Plotly charts
2. **Multi-panel Views**: Weight, quality, innovation
3. **Rejection Analysis**: Color-coded severity
4. **Batch Generation**: Parallel processing support

## Key Architectural Decisions

### 1. **Stateful Processing**
Each user maintains persistent Kalman state, enabling continuous learning and adaptation.

### 2. **Adaptive Noise Model**
Source-specific noise multipliers (0.5x for clinical, 3.0x for unreliable) improve accuracy.

### 3. **Quality Override System**
High-quality measurements (>0.8) bypass outlier detection, preventing false rejections.

### 4. **Reset Hierarchy**
Three reset types (INITIAL > HARD > SOFT) with different adaptation aggressiveness.

### 5. **Profile-Based Configuration**
High-level profiles (strict/moderate/lenient) simplify configuration while allowing overrides.

## Critical Integration Points

### 1. **Processor ↔ Database**
- State retrieval/persistence per measurement
- Atomic operations with rollback capability

### 2. **Processor ↔ Kalman**
- Adaptive parameter selection based on reset state
- Innovation calculation for outlier detection

### 3. **Quality Scorer ↔ Outlier Detector**
- Quality scores can override statistical outlier detection
- Protects high-confidence measurements

### 4. **Config Loader ↔ All Modules**
- Profile interpretation affects all processing parameters
- Cascading configuration from profiles to overrides

### 5. **Validation ↔ Constants**
- Hard-coded safety limits prevent dangerous configurations
- Physiological boundaries enforced regardless of config

## Data Source Hierarchy
1. **care-team-upload** (0.5x noise): Most reliable
2. **patient-upload** (0.7x noise): Very reliable
3. **questionnaire** (0.8x noise): Reliable
4. **patient-device** (1.0x noise): Standard
5. **connectivehealth.io** (1.5x noise): Less reliable
6. **iglucose.com** (3.0x noise): Requires validation

## Next Steps
1. Monitor reset frequency to validate gap thresholds
2. Analyze quality score distributions by source
3. Validate noise multipliers against real data

## Risks
- Complex state management could lead to corruption
- Adaptive parameters may over-correct after resets
- Quality override might accept true outliers
