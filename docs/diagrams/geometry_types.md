# Geometry Types

```mermaid
graph TD
    A[Geometry] --> B[Points]
    A --> C[Voxels]
    A --> D[Grid]
    A --> E[FactorGrid]

    B -->|neighbor search| F[PointConv]
    C -->|hash table kernel map| G[SparseConv3d]
    D -->|regular indexing| H[GridConv]
    E -->|factorized views| I[FactorGridConv]
```

Each geometry type pairs with specific convolution operations. `Points` uses
real-valued neighbor search (KNN or radius). `Voxels` builds a kernel map via
hash table lookup. `Grid` and `FactorGrid` use regular grid indexing.
