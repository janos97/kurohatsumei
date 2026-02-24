"""KuroHatsumei - Text-to-STL Pipeline for Resin 3D Printing.

A Gradio-based interface for generating 3D printable STL files from text descriptions.
"""

import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from src.config import get_config
from src.utils.file_manager import create_project_folder, load_json, save_json
from src.pipeline.stage1_prompt import generate_prompts, check_ollama
from src.pipeline.stage2_image import (
    generate_images, save_selected_image, check_comfyui,
    check_image_backend, get_available_image_backends,
)
from src.pipeline.stage3_mesh import generate_mesh, get_available_backends
from src.pipeline.stage4_repair import repair_and_export, format_stats_table


def check_services() -> str:
    """Check status of all required services."""
    lines = ["## Service Status\n"]

    # Check Ollama
    ok, msg = check_ollama()
    status = "✓" if ok else "✗"
    lines.append(f"- **Ollama**: {status} {msg}")

    # Check image backends
    image_backends = get_available_image_backends()
    lines.append("\n### Image Backends")
    for name, (ok, msg) in image_backends.items():
        status = "✓" if ok else "✗"
        lines.append(f"- **{name}**: {status} {msg}")

    # Check 3D backends
    backends = get_available_backends()
    lines.append("\n### 3D Backends")
    for name, (ok, msg) in backends.items():
        status = "✓" if ok else "✗"
        lines.append(f"- **{name}**: {status} {msg}")

    return "\n".join(lines)


# Stage 1: Prompt Engineering
def stage1_generate(description: str, project_state: dict) -> tuple[str, str, str, dict]:
    """Generate prompts from user description."""
    if not description.strip():
        return "", "", "Please enter a description", project_state

    # Create project folder if needed
    if not project_state.get("folder"):
        folder = create_project_folder(description[:30])
        project_state["folder"] = str(folder)

    success, result = generate_prompts(description, project_state["folder"])

    if success:
        project_state["prompts"] = result
        return (
            result["image_prompt"],
            result["threed_hint"],
            f"Prompts generated. Project: {Path(project_state['folder']).name}",
            project_state,
        )
    else:
        return "", "", f"Error: {result}", project_state


# Stage 2: Image Generation
def stage2_generate(
    image_prompt: str,
    num_images: int,
    image_backend: str,
    project_state: dict,
    progress=gr.Progress(),
) -> tuple[list[Image.Image], str, dict]:
    """Generate images from prompt."""
    if not image_prompt.strip():
        return [], "Please generate or enter an image prompt first", project_state

    if not project_state.get("folder"):
        folder = create_project_folder()
        project_state["folder"] = str(folder)

    def callback(msg: str) -> None:
        progress(0.5, desc=msg)

    success, result = generate_images(
        image_prompt,
        project_state["folder"],
        num_images=num_images,
        backend=image_backend.lower() if image_backend else None,
        callback=callback,
    )

    if success:
        project_state["images"] = [
            str(Path(project_state["folder"]) / f"generated_{i:02d}.png")
            for i in range(len(result))
        ]
        return result, f"Generated {len(result)} images", project_state
    else:
        return [], f"Error: {result}", project_state


def stage2_select(
    evt: gr.SelectData,
    gallery: list,
    project_state: dict,
) -> tuple[Image.Image | None, str, dict]:
    """Handle image selection from gallery."""
    if evt.index is None or not gallery:
        return None, "No image selected", project_state

    selected_img = gallery[evt.index]

    # Handle different gallery formats
    if isinstance(selected_img, tuple):
        selected_img = selected_img[0]
    if isinstance(selected_img, str):
        selected_img = Image.open(selected_img)

    if project_state.get("folder"):
        path = save_selected_image(selected_img, project_state["folder"])
        project_state["selected_image"] = str(path)
        return selected_img, f"Selected image {evt.index + 1}", project_state
    else:
        return selected_img, "Selected (no project folder)", project_state


def stage2_upload(
    image: Image.Image,
    project_state: dict,
) -> tuple[str, dict]:
    """Handle uploaded image."""
    if image is None:
        return "No image uploaded", project_state

    if not project_state.get("folder"):
        folder = create_project_folder()
        project_state["folder"] = str(folder)

    path = save_selected_image(image, project_state["folder"])
    project_state["selected_image"] = str(path)
    return f"Uploaded image saved to project", project_state


# Stage 3: 3D Generation
def stage3_generate(
    selected_image: Image.Image | None,
    backend: str,
    project_state: dict,
    progress=gr.Progress(),
) -> tuple[str | None, str, dict]:
    """Generate 3D mesh from selected image."""
    if selected_image is None:
        return None, "Please select or upload an image first", project_state

    if not project_state.get("folder"):
        folder = create_project_folder()
        project_state["folder"] = str(folder)

    # Save image if not already saved
    if not project_state.get("selected_image"):
        path = save_selected_image(selected_image, project_state["folder"])
        project_state["selected_image"] = str(path)

    def callback(msg: str) -> None:
        progress(0.5, desc=msg)

    success, result = generate_mesh(
        project_state["selected_image"],
        project_state["folder"],
        backend=backend.lower() if backend else None,
        callback=callback,
    )

    if success:
        # Save GLB for viewer
        glb_path = Path(project_state["folder"]) / "raw_mesh.glb"
        project_state["raw_mesh"] = str(glb_path)
        return str(glb_path), "3D mesh generated successfully", project_state
    else:
        return None, f"Error: {result}", project_state


def stage3_upload(
    mesh_file,
    project_state: dict,
) -> tuple[str | None, str, dict]:
    """Handle uploaded mesh file."""
    if mesh_file is None:
        return None, "No mesh uploaded", project_state

    if not project_state.get("folder"):
        folder = create_project_folder()
        project_state["folder"] = str(folder)

    # Copy the uploaded file
    dest_path = Path(project_state["folder"]) / "raw_mesh.glb"

    if hasattr(mesh_file, "name"):
        import shutil
        shutil.copy(mesh_file.name, dest_path)
    else:
        with open(dest_path, "wb") as f:
            f.write(mesh_file)

    project_state["raw_mesh"] = str(dest_path)
    return str(dest_path), "Mesh uploaded", project_state


# Stage 4: Repair & Export
def stage4_repair(
    close_boundaries: bool,
    project_state: dict,
    progress=gr.Progress(),
) -> tuple[str | None, str, str | None, str, dict]:
    """Repair mesh and export STL."""
    if not project_state.get("raw_mesh"):
        return None, "", None, "Please generate or upload a mesh first", project_state

    progress(0.3, desc="Repairing mesh...")

    success, result = repair_and_export(
        mesh_path=project_state["raw_mesh"],
        project_folder=project_state["folder"],
        close_boundaries=close_boundaries,
    )

    if success:
        stats_table = format_stats_table(result["stats"])
        glb_path = Path(project_state["folder"]) / "repaired_mesh.glb"
        stl_path = result["stl_path"]
        project_state["final_stl"] = str(stl_path)
        return str(glb_path), stats_table, str(stl_path), "Mesh repaired and ready!", project_state
    else:
        return None, "", None, f"Error: {result}", project_state


# Build the Gradio UI
def create_app() -> gr.Blocks:
    """Create the Gradio Blocks application."""
    config = get_config()

    with gr.Blocks(
        title="KuroHatsumei",
    ) as app:
        gr.Markdown("# KuroHatsumei")
        gr.Markdown("*Text-to-STL pipeline for resin 3D printing*")

        # Hidden state
        project_state = gr.State({})

        # Service status
        with gr.Accordion("Service Status", open=False):
            status_md = gr.Markdown(check_services())
            refresh_btn = gr.Button("Refresh Status", size="sm")
            refresh_btn.click(check_services, outputs=status_md)

        # Stage 1: Prompt Engineering
        with gr.Tab("1. Describe"):
            gr.Markdown("### Describe what you want to print")
            with gr.Row():
                with gr.Column(scale=2):
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="A small dragon figurine with wings spread...",
                        lines=3,
                    )
                    generate_prompt_btn = gr.Button("Generate Prompts", variant="primary")

                with gr.Column(scale=3):
                    image_prompt_output = gr.Textbox(
                        label="Image Prompt",
                        lines=4,
                        interactive=True,
                    )
                    threed_hint_output = gr.Textbox(
                        label="3D Hint",
                        lines=2,
                        interactive=False,
                    )

            stage1_status = gr.Markdown("")

            generate_prompt_btn.click(
                stage1_generate,
                inputs=[description_input, project_state],
                outputs=[image_prompt_output, threed_hint_output, stage1_status, project_state],
            )

        # Stage 2: Image Generation
        with gr.Tab("2. Generate Image"):
            gr.Markdown("### Generate or upload an image")

            with gr.Row():
                with gr.Column(scale=1):
                    image_backend_radio = gr.Radio(
                        choices=["sd_cpp", "comfyui"],
                        value=config.default_image_backend,
                        label="Image Backend",
                    )
                    num_images_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Number of images",
                    )
                    generate_images_btn = gr.Button("Generate Images", variant="primary")

                    gr.Markdown("**Or upload your own image:**")
                    upload_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                    )

                with gr.Column(scale=2):
                    image_gallery = gr.Gallery(
                        label="Generated Images (click to select)",
                        columns=2,
                        height=400,
                    )
                    selected_image_preview = gr.Image(
                        label="Selected Image",
                        height=200,
                    )

            stage2_status = gr.Markdown("")

            generate_images_btn.click(
                stage2_generate,
                inputs=[image_prompt_output, num_images_slider, image_backend_radio, project_state],
                outputs=[image_gallery, stage2_status, project_state],
            )

            image_gallery.select(
                stage2_select,
                inputs=[image_gallery, project_state],
                outputs=[selected_image_preview, stage2_status, project_state],
            )

            upload_image.change(
                stage2_upload,
                inputs=[upload_image, project_state],
                outputs=[stage2_status, project_state],
            ).then(
                lambda img: img,
                inputs=[upload_image],
                outputs=[selected_image_preview],
            )

        # Stage 3: 3D Generation
        with gr.Tab("3. Generate 3D"):
            gr.Markdown("### Convert image to 3D model")

            with gr.Row():
                with gr.Column(scale=1):
                    backend_radio = gr.Radio(
                        choices=["TripoSR", "TRELLIS", "Meshy"],
                        value="TripoSR",
                        label="3D Backend",
                    )
                    generate_3d_btn = gr.Button("Generate 3D", variant="primary")

                    gr.Markdown("**Or upload your own mesh:**")
                    upload_mesh = gr.File(
                        label="Upload Mesh (GLB/OBJ/STL)",
                        file_types=[".glb", ".obj", ".stl", ".ply"],
                    )

                with gr.Column(scale=2):
                    mesh_viewer = gr.Model3D(
                        label="Generated Mesh",
                        height=400,
                    )

            stage3_status = gr.Markdown("")

            generate_3d_btn.click(
                stage3_generate,
                inputs=[selected_image_preview, backend_radio, project_state],
                outputs=[mesh_viewer, stage3_status, project_state],
            )

            upload_mesh.change(
                stage3_upload,
                inputs=[upload_mesh, project_state],
                outputs=[mesh_viewer, stage3_status, project_state],
            )

        # Stage 4: Repair & Export
        with gr.Tab("4. Repair & Export"):
            gr.Markdown("### Repair mesh and export for printing")

            with gr.Row():
                with gr.Column(scale=1):
                    close_boundaries_cb = gr.Checkbox(
                        label="Close open boundaries",
                        value=False,
                        info="Use pymeshfix to close holes. May alter geometry.",
                    )
                    repair_btn = gr.Button("Repair & Export", variant="primary")

                    gr.Markdown("### Mesh Statistics")
                    stats_output = gr.Markdown("")

                    download_stl = gr.File(
                        label="Download STL",
                        visible=True,
                    )

                with gr.Column(scale=2):
                    repaired_mesh_viewer = gr.Model3D(
                        label="Repaired Mesh",
                        height=400,
                    )

            stage4_status = gr.Markdown("")

            repair_btn.click(
                stage4_repair,
                inputs=[close_boundaries_cb, project_state],
                outputs=[repaired_mesh_viewer, stats_output, download_stl, stage4_status, project_state],
            )

        # Footer
        gr.Markdown("---")
        gr.Markdown(
            "*Optimized for Anycubic Photon Mono M7 Pro "
            f"(build volume: {config.build_volume['x']}×{config.build_volume['y']}×{config.build_volume['z']}mm)*"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(theme=gr.themes.Soft())
