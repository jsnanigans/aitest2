# Reset System Architecture

## Reset Decision Flow

```
New Measurement Arrives
         ↓
┌─────────────────────┐
│  ResetManager       │
│  .should_trigger()  │
└─────────────────────┘
         ↓
    Check Type?
    ↙    ↓    ↘
Initial  Hard  Soft
   ↓      ↓     ↓
No       30+   Manual
Kalman   day   source
params   gap   + >5kg
   ↓      ↓     ↓
   └──────┴─────┘
         ↓
   Reset Triggered
         ↓
┌─────────────────────┐
│  Get Parameters     │
│  for Reset Type     │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Clear State &      │
│  Set Adaptation     │
└─────────────────────┘
         ↓
   Continue Processing
   with Adaptive Params
```

## Reset Types Comparison

| Aspect | Hard Reset | Initial Reset | Soft Reset |
|--------|------------|---------------|------------|
| **Trigger** | 30+ day gap | First measurement | Manual entry + >5kg change |
| **Weight Boost** | 10x | 10x | 3x |
| **Trend Boost** | 100x | 100x | 10x |
| **Decay Rate** | 3 (fast) | 3 (fast) | 5 (slow) |
| **Warmup Period** | 10 measurements | 10 measurements | 15 measurements |
| **Adaptive Days** | 7 days | 7 days | 10 days |
| **Quality Threshold** | 0.4 (very lenient) | 0.4 (very lenient) | 0.5 (moderately lenient) |
| **Use Case** | Long absence | New user | User correction |

## Manual Data Sources

```python
MANUAL_SOURCES = {
    # Questionnaires (highest trust)
    'internal-questionnaire'    # Regular check-ins
    'initial-questionnaire'      # Onboarding
    'questionnaire'             # Generic
    
    # User Uploads (high trust)
    'patient-upload'            # Patient entered
    'user-upload'              # User entered
    
    # Care Team (highest trust)
    'care-team-upload'         # Clinical entry
    'care-team-entry'          # Manual clinical
}
```

## Soft Reset Logic

```
IF source IN manual_sources:
    IF |current_weight - last_weight| >= 5kg:
        IF days_since_last_reset > 3:
            → TRIGGER SOFT RESET
```

## Parameter Evolution Over Time

### Hard/Initial Reset (Aggressive Adaptation)
```
Boost Factor
10x |****
    |   ****
    |      ****
    |         ****
1x  |____________****___
    0    3    6    10 measurements
    
Fast decay (rate=3) → Normal in ~10 measurements
```

### Soft Reset (Gentle Adaptation)
```
Boost Factor
3x  |********
    |       ********
    |              ********
1x  |_______________________********___
    0    5    10   15   20 measurements
    
Slow decay (rate=5) → Normal in ~15 measurements
```

## Quality Scoring During Reset

### Hard Reset Quality
- Threshold: 0.4 (accept most things)
- Focus: Get data flowing
- Duration: 7 days

### Soft Reset Quality  
- Threshold: 0.5 (moderate)
- Focus: Trust but verify
- Duration: 10 days

## State Tracking

```json
{
  "reset_events": [
    {
      "timestamp": "2025-01-15T10:00:00",
      "type": "soft",
      "trigger": "manual_entry",
      "source": "care-team-upload",
      "weight_change": 6.5,
      "parameters": {
        "weight_boost": 3,
        "trend_boost": 10,
        "decay_rate": 5
      }
    }
  ],
  "reset_type": "soft",
  "measurements_since_reset": 5,
  "reset_timestamp": "2025-01-15T10:00:00"
}
```

## Benefits of Parameterized System

1. **Contextual Adaptation**
   - Different scenarios need different approaches
   - Manual data is more trustworthy

2. **Prevent Over-correction**
   - Soft reset doesn't swing as wildly
   - Maintains more stability

3. **Better User Experience**
   - Accepts manual corrections gracefully
   - Doesn't punish users for updating weight

4. **Clinical Relevance**
   - Care team entries are trusted
   - System adapts to clinical updates

5. **Maintainability**
   - All reset logic in one place
   - Easy to tune parameters
   - Clear configuration structure