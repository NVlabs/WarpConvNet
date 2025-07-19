# Quick Start

After installation you can quickly verify the package with the ModelNet example:

```bash
python examples/modelnet.py
```

For ScanNet semantic segmentation:

```bash
pip install warpconvnet[models]
python examples/scannet.py train.batch_size=12 model=mink_unet
```
