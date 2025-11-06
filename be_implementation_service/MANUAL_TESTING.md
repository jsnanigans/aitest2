# Manual Testing Guide

## Prerequisites

Start the local SAM server:
```bash
sam local start-api --port 3080
```

## Test Process Endpoint

### Using Python (recommended)

```bash
python3 -c "
import json
import urllib.request

# Load the fixture
with open('tests/fixtures/process_request.json') as f:
    data = json.load(f)

# Extract the user_id and body
user_id = data['user_id']
body = data['body']

# Prepare the request
url = f'http://127.0.0.1:3080/api/v1/process/{user_id}'
headers = {'Content-Type': 'application/json'}
json_data = json.dumps(body).encode('utf-8')

# Make the request
req = urllib.request.Request(url, data=json_data, headers=headers, method='POST')
try:
    with urllib.request.urlopen(req) as response:
        print(f'Status: {response.status}')
        print(f'Response: {response.read().decode()}')
except urllib.error.HTTPError as e:
    print(f'Error: {e.code}')
    print(f'Response: {e.read().decode()}')
"
```

### Pretty-printed JSON output

```bash
python3 -c "
import json
import urllib.request

with open('tests/fixtures/process_request.json') as f:
    data = json.load(f)

user_id = data['user_id']
body = data['body']

url = f'http://127.0.0.1:3080/api/v1/process/{user_id}'
headers = {'Content-Type': 'application/json'}
json_data = json.dumps(body).encode('utf-8')

req = urllib.request.Request(url, data=json_data, headers=headers, method='POST')
try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f'Error: {e.code}')
    print(e.read().decode())
"
```

## Test Other Endpoints

### Health Check

```bash
python3 -c "
import urllib.request
response = urllib.request.urlopen('http://127.0.0.1:3080/api/v1/health')
print(response.read().decode())
"
```

### Get State

```bash
python3 -c "
import urllib.request
user_id = '5845fbd9-e241-4255-a877-84dee539a521'
url = f'http://127.0.0.1:3080/api/v1/state/{user_id}'
response = urllib.request.urlopen(url)
print(response.read().decode())
"
```

### Cleanup State

```bash
python3 -c "
import json
import urllib.request

user_id = '5845fbd9-e241-4255-a877-84dee539a521'
url = f'http://127.0.0.1:3080/api/v1/cleanup/{user_id}'
headers = {'Content-Type': 'application/json'}
json_data = json.dumps({}).encode('utf-8')

req = urllib.request.Request(url, data=json_data, headers=headers, method='POST')
try:
    with urllib.request.urlopen(req) as response:
        print(f'Status: {response.status}')
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(f'Error: {e.code}')
    print(e.read().decode())
"
```

### Replay Measurements

```bash
python3 -c "
import json
import urllib.request

user_id = '5845fbd9-e241-4255-a877-84dee539a521'
url = f'http://127.0.0.1:3080/api/v1/replay/{user_id}'
headers = {'Content-Type': 'application/json'}

# Optional: specify date range
body = {
    'from_date': '2025-01-01T00:00:00',
    'to_date': '2025-01-07T23:59:59'
}
json_data = json.dumps(body).encode('utf-8')

req = urllib.request.Request(url, data=json_data, headers=headers, method='POST')
try:
    with urllib.request.urlopen(req) as response:
        print(f'Status: {response.status}')
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(f'Error: {e.code}')
    print(e.read().decode())
"
```

## Test User ID

The fixture uses this user ID:
```
5845fbd9-e241-4255-a877-84dee539a521
```

## Expected Results from Fixture

When processing the fixture data:
- Total measurements: 15
- Accepted: 11
- Rejected: 4

Rejected measurements are filtered by quality scoring based on anomaly detection and Kalman fit.
