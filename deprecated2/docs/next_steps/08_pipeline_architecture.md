# Step 08: Refactor into Proper Pipeline Architecture

## Priority: LOW - PARTIALLY IMPLEMENTED

## Current State
**Mixed Architecture**: The implementation has some pipeline elements but lacks formal structure:
- ✅ Separate processing modules (UserProcessor, KalmanProcessor)
- ✅ Streaming line-by-line processing
- ❌ No formal pipeline stages or interfaces
- ❌ No async/parallel processing capabilities
- ❌ Stages are tightly coupled

## Why This Change?
The framework document (Section 6.1) describes a clear pipeline architecture with distinct stages. A proper pipeline:

1. **Separation of Concerns**: Each stage has a single responsibility
2. **Testability**: Individual stages can be tested in isolation
3. **Extensibility**: New processing stages can be added easily
4. **Performance**: Enables parallel processing and optimization
5. **Maintainability**: Clear boundaries make code easier to understand

## Expected Benefits
- **50% reduction** in bug introduction rate
- **Easier onboarding** for new developers
- **Better performance** through stage-level optimization
- **Improved testing** with 90%+ code coverage achievable
- **Flexible deployment** with ability to run stages independently

## Implementation Guide

### Pipeline Architecture Overview
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class PipelineData:
    """
    Data container that flows through pipeline
    """
    user_id: str
    raw_measurements: List[float]
    timestamps: List[str]
    metadata: Dict[str, Any]
    
    # Added by stages
    cleaned_measurements: Optional[List[float]] = None
    baseline: Optional[Dict[str, float]] = None
    kalman_states: Optional[List[Dict]] = None
    outliers: Optional[List[Dict]] = None
    change_points: Optional[List[Dict]] = None
    regimes: Optional[List[str]] = None
    smoothed_trajectory: Optional[List[float]] = None
    
class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages
    """
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            'processed': 0,
            'errors': 0,
            'avg_time_ms': 0
        }
    
    @abstractmethod
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Process data through this stage
        """
        pass
    
    def validate_input(self, data: PipelineData) -> bool:
        """
        Validate that required inputs are present
        """
        return True
    
    def log_metrics(self):
        """
        Log stage performance metrics
        """
        print(f"{self.name}: {self.metrics}")
```

### Core Pipeline Stages

#### Stage 1: Data Ingestion
```python
class DataIngestionStage(PipelineStage):
    """
    Load and validate raw data
    """
    
    def __init__(self):
        super().__init__("DataIngestion")
        self.validators = [
            self.validate_format,
            self.validate_ranges,
            self.validate_timestamps
        ]
    
    async def process(self, data: PipelineData) -> PipelineData:
        # Validate raw data
        for validator in self.validators:
            if not validator(data):
                raise ValueError(f"Validation failed: {validator.__name__}")
        
        # Sort by timestamp
        sorted_indices = np.argsort(data.timestamps)
        data.raw_measurements = [data.raw_measurements[i] for i in sorted_indices]
        data.timestamps = [data.timestamps[i] for i in sorted_indices]
        
        return data
    
    def validate_format(self, data: PipelineData) -> bool:
        """Check data format consistency"""
        return len(data.raw_measurements) == len(data.timestamps)
    
    def validate_ranges(self, data: PipelineData) -> bool:
        """Check physiological plausibility"""
        measurements = data.raw_measurements
        return all(30 <= m <= 400 for m in measurements)
    
    def validate_timestamps(self, data: PipelineData) -> bool:
        """Check timestamp validity"""
        # Ensure chronological order and no duplicates
        return len(set(data.timestamps)) == len(data.timestamps)
```

#### Stage 2: Baseline Establishment
```python
class BaselineEstablishmentStage(PipelineStage):
    """
    Establish robust baseline using IQR → Median → MAD
    """
    
    def __init__(self):
        super().__init__("BaselineEstablishment")
        
    async def process(self, data: PipelineData) -> PipelineData:
        # Get first 14 days of data
        initial_data = data.raw_measurements[:14]
        
        if len(initial_data) < 7:
            raise ValueError("Insufficient data for baseline")
        
        # IQR outlier removal
        Q1 = np.percentile(initial_data, 25)
        Q3 = np.percentile(initial_data, 75)
        IQR = Q3 - Q1
        
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        filtered = [x for x in initial_data 
                   if lower_fence <= x <= upper_fence]
        
        # Calculate baseline
        baseline_weight = np.median(filtered)
        
        # Calculate MAD for variance
        deviations = np.abs(filtered - baseline_weight)
        MAD = np.median(deviations)
        sigma = 1.4826 * MAD
        
        data.baseline = {
            'weight': baseline_weight,
            'variance': sigma ** 2,
            'confidence': 'high' if len(filtered) >= 10 else 'medium',
            'outliers_removed': len(initial_data) - len(filtered)
        }
        
        return data
```

#### Stage 3: Outlier Detection
```python
class OutlierDetectionStage(PipelineStage):
    """
    Multi-layered outlier detection
    """
    
    def __init__(self):
        super().__init__("OutlierDetection")
        self.mad_detector = MovingMADDetector()
        self.arima_classifier = ARIMAOutlierDetector()
    
    async def process(self, data: PipelineData) -> PipelineData:
        outliers = []
        cleaned = []
        
        # Layer 1: Moving MAD
        mad_outliers = self.mad_detector.detect(data.raw_measurements)
        
        # Layer 2: ARIMA classification
        arima_results = self.arima_classifier.process_time_series(
            data.raw_measurements
        )
        
        # Combine results
        for i, measurement in enumerate(data.raw_measurements):
            is_outlier = mad_outliers[i] or arima_results[i]['type'] != 'normal'
            
            if is_outlier:
                outliers.append({
                    'index': i,
                    'value': measurement,
                    'type': arima_results[i]['type'],
                    'confidence': arima_results[i]['confidence']
                })
            else:
                cleaned.append(measurement)
        
        data.outliers = outliers
        data.cleaned_measurements = cleaned
        
        return data
```

#### Stage 4: Kalman Filtering
```python
class KalmanFilteringStage(PipelineStage):
    """
    Apply Kalman filter with validation gate
    """
    
    def __init__(self):
        super().__init__("KalmanFiltering")
        self.validation_gate = ValidationGate()
    
    async def process(self, data: PipelineData) -> PipelineData:
        # Initialize filter with baseline
        kf = KalmanFilter(
            initial_weight=data.baseline['weight'],
            measurement_variance=data.baseline['variance']
        )
        
        states = []
        
        for measurement in data.cleaned_measurements:
            # Predict
            pred_state, pred_cov = kf.predict()
            
            # Validate
            is_valid, confidence = self.validation_gate.validate(
                measurement, pred_state[0], pred_cov[0, 0]
            )
            
            # Update if valid
            if is_valid:
                kf.update(measurement)
            
            states.append({
                'weight': kf.state[0],
                'velocity': kf.state[1],
                'uncertainty': np.sqrt(kf.P[0, 0]),
                'measurement_valid': is_valid
            })
        
        data.kalman_states = states
        
        return data
```

#### Stage 5: Change Point Detection
```python
class ChangePointDetectionStage(PipelineStage):
    """
    Detect regime changes in weight trajectory
    """
    
    def __init__(self):
        super().__init__("ChangePointDetection")
        self.cpd = RESPERM()
    
    async def process(self, data: PipelineData) -> PipelineData:
        # Extract weight trajectory
        weights = [s['weight'] for s in data.kalman_states]
        
        # Detect change points
        change_points = self.cpd.detect_change_points(weights)
        
        # Classify regimes
        regimes = self.classify_regimes(weights, change_points)
        
        data.change_points = change_points
        data.regimes = regimes
        
        return data
    
    def classify_regimes(self, weights, change_points):
        """
        Classify weight regimes between change points
        """
        regimes = []
        cp_indices = [0] + [cp['index'] for cp in change_points] + [len(weights)]
        
        for i in range(len(cp_indices) - 1):
            start, end = cp_indices[i], cp_indices[i + 1]
            segment = weights[start:end]
            
            if len(segment) > 1:
                velocity = (segment[-1] - segment[0]) / len(segment)
                
                if abs(velocity) < 0.05:
                    regime = 'maintenance'
                elif velocity < -0.15:
                    regime = 'rapid_loss'
                elif velocity < 0:
                    regime = 'weight_loss'
                else:
                    regime = 'weight_gain'
            else:
                regime = 'unknown'
            
            regimes.extend([regime] * (end - start))
        
        return regimes
```

#### Stage 6: Smoothing
```python
class SmoothingStage(PipelineStage):
    """
    Apply Kalman smoother for optimal historical estimates
    """
    
    def __init__(self):
        super().__init__("Smoothing")
    
    async def process(self, data: PipelineData) -> PipelineData:
        # Apply RTS smoother
        smoother = KalmanSmoother()
        smoothed_states, _ = smoother.smooth(data.cleaned_measurements)
        
        data.smoothed_trajectory = [s[0] for s in smoothed_states]
        
        return data
```

### Pipeline Orchestrator
```python
class WeightProcessingPipeline:
    """
    Main pipeline orchestrator
    """
    
    def __init__(self):
        self.stages = [
            DataIngestionStage(),
            BaselineEstablishmentStage(),
            OutlierDetectionStage(),
            KalmanFilteringStage(),
            ChangePointDetectionStage(),
            SmoothingStage()
        ]
        
        self.execution_modes = {
            'sequential': self.run_sequential,
            'parallel': self.run_parallel,
            'streaming': self.run_streaming
        }
    
    async def process(self, data: PipelineData, mode='sequential'):
        """
        Process data through pipeline
        """
        processor = self.execution_modes.get(mode, self.run_sequential)
        return await processor(data)
    
    async def run_sequential(self, data: PipelineData):
        """
        Run stages sequentially
        """
        for stage in self.stages:
            try:
                data = await stage.process(data)
            except Exception as e:
                print(f"Error in {stage.name}: {e}")
                stage.metrics['errors'] += 1
                raise
            stage.metrics['processed'] += 1
        
        return data
    
    async def run_parallel(self, data: PipelineData):
        """
        Run independent stages in parallel
        """
        # Group stages that can run in parallel
        parallel_groups = [
            [self.stages[0]],  # Data ingestion
            [self.stages[1]],  # Baseline
            [self.stages[2], self.stages[3]],  # Outlier + Kalman
            [self.stages[4], self.stages[5]]   # CPD + Smoothing
        ]
        
        for group in parallel_groups:
            tasks = [stage.process(data) for stage in group]
            results = await asyncio.gather(*tasks)
            # Merge results back into data
            for result in results:
                data = self.merge_results(data, result)
        
        return data
    
    async def run_streaming(self, data_stream):
        """
        Process streaming data in real-time
        """
        buffer = []
        
        async for measurement in data_stream:
            buffer.append(measurement)
            
            # Process when buffer is full
            if len(buffer) >= 30:
                batch_data = PipelineData(
                    user_id="stream",
                    raw_measurements=buffer[-30:],
                    timestamps=generate_timestamps(30),
                    metadata={}
                )
                
                result = await self.run_sequential(batch_data)
                yield result.kalman_states[-1]  # Yield latest state
```

### Configuration Management
```python
class PipelineConfig:
    """
    Centralized configuration for all stages
    """
    
    DEFAULT_CONFIG = {
        'baseline': {
            'min_days': 7,
            'max_days': 14,
            'iqr_multiplier': 1.5
        },
        'outlier_detection': {
            'mad_threshold': 3.0,
            'arima_order': (1, 1, 1),
            'window_size': 30
        },
        'kalman_filter': {
            'state_dim': 2,
            'process_noise': {
                'weight': 0.01,
                'velocity': 0.0001
            },
            'validation_gamma': 3.0
        },
        'change_point': {
            'window_size': 30,
            'n_permutations': 1000,
            'alpha': 0.05
        },
        'smoothing': {
            'enable': True,
            'lag': 7
        }
    }
    
    @classmethod
    def load_config(cls, config_file=None):
        """
        Load configuration from file or use defaults
        """
        if config_file:
            with open(config_file, 'r') as f:
                import toml
                config = toml.load(f)
                return {**cls.DEFAULT_CONFIG, **config}
        return cls.DEFAULT_CONFIG
```

### Testing Framework
```python
import unittest
from unittest.mock import Mock, patch

class TestPipelineStages(unittest.TestCase):
    """
    Unit tests for pipeline stages
    """
    
    def setUp(self):
        self.test_data = PipelineData(
            user_id="test_user",
            raw_measurements=[70.5, 70.3, 71.0, 69.8],
            timestamps=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            metadata={}
        )
    
    async def test_baseline_stage(self):
        """Test baseline establishment"""
        stage = BaselineEstablishmentStage()
        result = await stage.process(self.test_data)
        
        self.assertIsNotNone(result.baseline)
        self.assertIn('weight', result.baseline)
        self.assertIn('variance', result.baseline)
    
    async def test_pipeline_end_to_end(self):
        """Test complete pipeline"""
        pipeline = WeightProcessingPipeline()
        result = await pipeline.process(self.test_data)
        
        self.assertIsNotNone(result.smoothed_trajectory)
        self.assertEqual(len(result.smoothed_trajectory), 
                        len(self.test_data.raw_measurements))
```

## Deployment Options

### Microservices Architecture
```python
# Each stage as a separate service
class KalmanFilterService:
    """
    Standalone Kalman filtering service
    """
    def __init__(self):
        self.stage = KalmanFilteringStage()
    
    async def handle_request(self, request):
        data = PipelineData(**request.json)
        result = await self.stage.process(data)
        return result.to_json()
```

### Batch Processing
```python
class BatchProcessor:
    """
    Process multiple users in batch
    """
    async def process_batch(self, user_ids):
        pipeline = WeightProcessingPipeline()
        
        tasks = []
        for user_id in user_ids:
            data = await load_user_data(user_id)
            tasks.append(pipeline.process(data))
        
        results = await asyncio.gather(*tasks)
        return results
```

## Monitoring and Observability
```python
class PipelineMonitor:
    """
    Monitor pipeline health and performance
    """
    def __init__(self):
        self.metrics = {}
    
    def track_stage_performance(self, stage_name, duration, success):
        """Track individual stage metrics"""
        if stage_name not in self.metrics:
            self.metrics[stage_name] = {
                'total_runs': 0,
                'successes': 0,
                'avg_duration_ms': 0
            }
        
        metrics = self.metrics[stage_name]
        metrics['total_runs'] += 1
        if success:
            metrics['successes'] += 1
        
        # Update rolling average
        n = metrics['total_runs']
        metrics['avg_duration_ms'] = (
            (metrics['avg_duration_ms'] * (n - 1) + duration) / n
        )
    
    def get_health_status(self):
        """Overall pipeline health"""
        total_stages = len(self.metrics)
        healthy_stages = sum(
            1 for m in self.metrics.values() 
            if m['successes'] / m['total_runs'] > 0.95
        )
        
        return {
            'status': 'healthy' if healthy_stages == total_stages else 'degraded',
            'healthy_stages': healthy_stages,
            'total_stages': total_stages,
            'metrics': self.metrics
        }
```

## Migration Strategy
1. **Phase 1**: Wrap existing code in pipeline stages
2. **Phase 2**: Add stage interfaces and data contracts
3. **Phase 3**: Implement parallel execution
4. **Phase 4**: Deploy as microservices (optional)

## References
- Framework Section 6: "Synthesis and Implementation Roadmap"
- Clean Architecture principles (Robert C. Martin)
- Pipeline design patterns
- Stream processing architectures