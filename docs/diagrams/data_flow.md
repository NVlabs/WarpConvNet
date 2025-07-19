```mermaid
flowchart LR
    A[Raw Points] --> B[Coordinate Processing]
    B --> C[Feature Processing]
    C --> D[Neural Network]
    D --> E[Output]
    
    subgraph "Geometry Pipeline"
        B
        C
    end
    
    subgraph "Network Pipeline"
        D
    end
```