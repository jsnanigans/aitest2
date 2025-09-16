# Configuration Guide

## Quick Start

Instead of editing the complex `config.toml` directly, use the new **profile-based configuration system**:

1. Edit `config_profiles.toml`
2. Choose a profile: `conservative`, `balanced`, `aggressive`, or `clinical`
3. Run: `uv run python src/config_generator.py`

This generates a complete `config.toml` with all the appropriate parameters.

## Understanding Profiles

### Conservative Profile
- **Use when:** You need high confidence in data quality
- **Behavior:** Strict filtering, slow to adapt to weight changes, rejects questionable data
- **Good for:** Clinical trials, medical monitoring

### Balanced Profile (Default)
- **Use when:** Standard weight tracking scenarios
- **Behavior:** Moderate filtering, reasonable adaptation speed
- **Good for:** General health monitoring, most users

### Aggressive Profile
- **Use when:** You want to capture all weight changes quickly
- **Behavior:** Minimal filtering, quick adaptation, accepts most data
- **Good for:** Athletes, users with rapid weight changes

### Clinical Profile
- **Use when:** In healthcare settings with mixed data sources
- **Behavior:** Trusts clinical sources highly, stricter on consumer devices
- **Good for:** Hospital/clinic deployments

## Simple Configuration

Edit only these fields in `config_profiles.toml`:

```toml
active_profile = "balanced"  # Choose your profile

[user_settings]
input_file = "./data/weights.csv"
output_directory = "output"
max_users_to_process = 200
minimum_readings_required = 20
generate_charts = true
chart_detail_level = "normal"  # "minimal", "normal", or "detailed"
```

## What Each Setting Controls

### Profile Effects

| Setting | Conservative | Balanced | Aggressive |
|---------|-------------|----------|------------|
| Outlier threshold | ±10% | ±15% | ±25% |
| Quality required | 70% | 60% | 45% |
| Adaptation speed | 20 measurements | 10 measurements | 5 measurements |
| Gap sensitivity | 21 days | 30 days | 45 days |

### Advanced Overrides (Optional)

Only change these if profiles don't meet your needs:

- `extreme_deviation_percent`: Maximum % a weight can deviate (default: 15)
- `gap_threshold_days`: Days before data is considered "stale" (default: 30)
- `manual_entry_change_threshold_kg`: Weight change to trigger adaptation (default: 5kg)

## Command Line Usage

```bash
# Generate config with current profile
uv run python src/config_generator.py

# Use specific profiles file
uv run python src/config_generator.py --profiles my_profiles.toml

# Preview without saving
uv run python src/config_generator.py --show

# Process data with generated config
uv run python main.py data/weights.csv
```

## Migration from Old Config

Your existing `config.toml` still works! The new system is optional. To migrate:

1. Back up your current `config.toml`
2. Choose the closest profile to your current settings
3. Add any special overrides to `[advanced]` section
4. Generate and test the new config

## Understanding the Effects

### "Why is my data being rejected?"
- Try switching from `conservative` to `balanced`
- Lower the quality threshold in advanced settings
- Check if you have large gaps triggering resets

### "Why does it take so long to adapt to weight loss?"
- Switch from `conservative` to `aggressive`
- Reduce adaptation measurements in advanced settings

### "Why is noisy data getting through?"
- Switch from `aggressive` to `conservative`
- Increase the quality threshold
- Decrease extreme_deviation_percent

## Full Parameter Mapping

The profile system manages these complex parameters for you:

- **Kalman filter parameters** (4 base + 36 reset parameters)
- **Quality scoring weights** (4 components × 3 contexts)
- **Adaptation speeds** (3 reset types × 4 parameters)
- **Visualization settings** (19 display options)
- **Retrospective processing** (9 parameters)

Total: ~80 parameters reduced to 1 profile choice + 6 simple settings!