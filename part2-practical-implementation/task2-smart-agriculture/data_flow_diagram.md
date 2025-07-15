```mermaid
flowchart TB
    A[Soil Moisture Sensor] --> B[Raspberry Pi]
    C[Temperature Sensor] --> B
    D[Camera] --> B
    B --> E{AI Model}
    E -->|Prediction| F[Cloud Dashboard]
    E -->|Alert| G[Farmer's Phone]
```
+-------------------+       +---------------------+       +-------------------+  
|    IoT Sensors    | ----> | Edge Device (RPi)   | ----> | Cloud/Server      |  
+-------------------+       +---------------------+       +-------------------+  
  │  │  │  │                   │                            │  
  │  │  │  └─ Soil Moisture    │                            └─ Dashboard (Web/Mobile)  
  │  │  └─ Temperature         │  
  │  └─ Humidity               └── AI Model (LSTM/CNN) ───> Farmer Alerts (SMS)  
  └─ Camera (Pest Detection)  
