```mermaid
flowchart TB
    A[Soil Moisture Sensor] --> B[Raspberry Pi]
    C[Temperature Sensor] --> B
    D[Camera] --> B
    B --> E{AI Model}
    E -->|Prediction| F[Cloud Dashboard]
    E -->|Alert| G[Farmer's Phone]
```
