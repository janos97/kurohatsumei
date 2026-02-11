# KuroHatsumei

Text-to-STL pipeline for resin 3D printing. Generate 3D printable models from text descriptions using AI.

## Overview

KuroHatsumei is a 4-stage pipeline that transforms text descriptions into print-ready STL files:

1. **Prompt Engineering** - Uses Ollama to generate optimized prompts for 3D-printable image generation
2. **Image Generation** - Creates reference images via ComfyUI with FLUX
3. **3D Generation** - Converts images to 3D meshes using TripoSR, TRELLIS, or Meshy
4. **Mesh Repair & Export** - Repairs geometry and exports STL scaled for your printer

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with a model (default: `llama3.1:8b`)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with FLUX model
- One of:
  - [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (local, requires GPU)
  - [TRELLIS](https://github.com/microsoft/TRELLIS) (local, requires GPU with 12GB+ VRAM)
  - [Meshy](https://www.meshy.ai/) API key (cloud-based)

## Installation

```bash
git clone https://github.com/janos97/kurohatsumei.git
cd kurohatsumei
pip install -r requirements.txt
```

For local 3D generation (optional):
```bash
# TripoSR
pip install git+https://github.com/VAST-AI-Research/TripoSR.git

# TRELLIS - see https://github.com/microsoft/TRELLIS for installation
```

## Configuration

Edit `config.yaml` or use environment variables:

```yaml
ollama:
  url: "http://localhost:11434"
  model: "llama3.1:8b"

comfyui:
  url: "http://localhost:8188"
  workflow_path: "workflows/flux_txt2img.json"
  num_images: 4

meshy:
  api_key: ""  # Or set MESHY_API_KEY env var

build_volume:  # Anycubic Photon Mono M7 Pro
  x: 223
  y: 126
  z: 230
  margin: 5

default_3d_backend: "triposr"  # triposr | trellis | meshy
```

Environment variables:
- `OLLAMA_URL` - Ollama server URL
- `COMFYUI_URL` - ComfyUI server URL
- `MESHY_API_KEY` - Meshy API key
- `KUROHATSUMEI_3D_BACKEND` - Default 3D backend

## Usage

```bash
python app.py
```

Open http://localhost:7860 in your browser.

### Workflow

1. **Describe** - Enter what you want to print (e.g., "a small dragon figurine with wings spread")
2. **Generate Image** - Create reference images, click to select one
3. **Generate 3D** - Convert the image to a 3D mesh
4. **Repair & Export** - Fix geometry issues and download the STL

You can skip stages by uploading your own images or meshes.

## Project Structure

```
kurohatsumei/
├── app.py                    # Gradio UI
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── workflows/
│   └── flux_txt2img.json     # ComfyUI workflow
└── src/
    ├── config.py             # Config loader
    ├── pipeline/
    │   ├── stage1_prompt.py  # Prompt engineering
    │   ├── stage2_image.py   # Image generation
    │   ├── stage3_mesh.py    # 3D generation
    │   └── stage4_repair.py  # Mesh repair
    ├── services/
    │   ├── ollama.py         # Ollama client
    │   ├── comfyui.py        # ComfyUI client
    │   ├── meshy.py          # Meshy client
    │   ├── triposr.py        # TripoSR wrapper
    │   └── trellis.py        # TRELLIS wrapper
    └── utils/
        ├── file_manager.py   # Project folders
        └── mesh_utils.py     # Mesh processing
```

## Output

Each generation creates a timestamped project folder in `output/` containing:
- `prompts.json` - Generated prompts
- `generated_*.png` - Generated images
- `selected_image.png` - Selected reference image
- `raw_mesh.glb` - Raw 3D mesh
- `repaired_mesh.glb` - Repaired mesh
- `final_print.stl` - Print-ready STL

## Mesh Repair

The repair stage automatically:
- Fixes inverted normals
- Removes degenerate/duplicate faces
- Keeps only the largest connected component (removes debris)
- Scales to fit build volume
- Centers on build plate

Optional: Enable "Close open boundaries" to use pymeshfix for watertight meshes (may alter geometry).

## License

MIT
