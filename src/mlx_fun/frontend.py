"""Gradio web frontend for the MLX-FUN server.

Connects to a running REAP server and provides:
- Chat interface (via OpenAI-compatible /v1/chat/completions)
- Expert activation dashboard (heatmaps from /v1/reap/stats)
- Steering controls (via /v1/reap/steer)
- Server management (save, reset, info)
- File comparison and merge mode analysis with dynamic filtering
"""

import csv
import io
import json
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for the frontend. Install with: "
        "pip install 'mlx-fun[ui]'"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    raise ImportError(
        "Matplotlib is required for the frontend. Install with: "
        "pip install 'mlx-fun[ui]'"
    )

import requests


# ---------------------------------------------------------------------------
# LocalStorage persistence JavaScript
# ---------------------------------------------------------------------------

def get_local_storage_script() -> str:
    """Return JavaScript for localStorage persistence of file names and filters.
    
    This script:
    1. Saves original file names when files are selected via gr.File()
    2. Displays previously selected file names as hints on page reload
    3. Persists filter values (min/max rank, n to prune)
    """
    return """
    <script>
    (function() {
        // Storage keys for file names and filters
        const FILE_STORAGE_KEYS = {
            'merge_file1': 'mlx_fun_merge_file1',
            'merge_file2': 'mlx_fun_merge_file2',
            'merge_file3': 'mlx_fun_merge_file3',
            'merge_file4': 'mlx_fun_merge_file4',
            'diff_file1': 'mlx_fun_diff_file1',
            'diff_file2': 'mlx_fun_diff_file2'
        };
        
        const FILTER_STORAGE_KEYS = {
            'rank_min_filter': 'mlx_fun_filter_min_rank',
            'rank_max_filter': 'mlx_fun_filter_max_rank',
            'top_n_filter': 'mlx_fun_filter_top_n'
        };
        
        function saveToStorage(key, value) {
            try {
                localStorage.setItem(key, value || '');
            } catch (e) {
                console.warn('localStorage save failed:', e);
            }
        }
        
        function loadFromStorage(key) {
            try {
                return localStorage.getItem(key) || '';
            } catch (e) {
                console.warn('localStorage load failed:', e);
                return '';
            }
        }
        
        function findFileInput(elemId) {
            // Find the file input container by elem_id
            let container = document.getElementById(elemId);
            if (!container) {
                container = document.querySelector(`[data-testid="${elemId}"]`);
            }
            if (!container) {
                // Try finding by id in shadow DOM-like structure
                container = document.querySelector(`#${elemId}`);
            }
            return container;
        }
        
        function findFilterInput(elemId) {
            // Try direct ID first
            let container = document.getElementById(elemId);
            if (container) {
                const input = container.querySelector('input[type="text"], input[type="number"], textarea');
                if (input) return input;
            }
            // Try data-testid
            let testContainer = document.querySelector(`[data-testid="${elemId}"]`);
            if (testContainer) {
                const input = testContainer.querySelector('input[type="text"], input[type="number"], textarea');
                if (input) return input;
            }
            return null;
        }
        
        function setupFilePersistence() {
            // Handle file inputs - save original filename when file is selected
            Object.entries(FILE_STORAGE_KEYS).forEach(([elemId, storageKey]) => {
                const container = findFileInput(elemId);
                if (container) {
                    // Find the file input element within the container
                    const fileInput = container.querySelector('input[type="file"]');
                    if (fileInput) {
                        fileInput.addEventListener('change', (e) => {
                            const files = e.target.files;
                            if (files && files.length > 0) {
                                // Save the original filename
                                const fileName = files[0].name;
                                saveToStorage(storageKey, fileName);
                                updateSavedFilesDisplay();
                            }
                        });
                    }
                }
            });
        }
        
        function setupFilterPersistence() {
            Object.entries(FILTER_STORAGE_KEYS).forEach(([elemId, storageKey]) => {
                const input = findFilterInput(elemId);
                if (input) {
                    // Load saved value
                    const saved = loadFromStorage(storageKey);
                    if (saved && !input.value) {
                        input.value = saved;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    
                    // Save on change
                    input.addEventListener('input', () => {
                        saveToStorage(storageKey, input.value);
                    });
                }
            });
        }
        
        function updateSavedFilesDisplay() {
            // Update the saved files display element if it exists
            const displayEl = document.getElementById('saved_files_display');
            if (!displayEl) return;
            
            const savedFiles = [];
            Object.entries(FILE_STORAGE_KEYS).forEach(([elemId, storageKey]) => {
                const fileName = loadFromStorage(storageKey);
                if (fileName) {
                    savedFiles.push(`${elemId}: ${fileName}`);
                }
            });
            
            if (savedFiles.length > 0) {
                displayEl.innerHTML = '<strong>Previously selected files:</strong><br>' +
                    savedFiles.map(f => '• ' + f).join('<br>');
            } else {
                displayEl.innerHTML = '';
            }
        }
        
        function setupPersistence() {
            setupFilePersistence();
            setupFilterPersistence();
            updateSavedFilesDisplay();
        }
        
        // Run after a short delay to ensure Gradio has rendered
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => setTimeout(setupPersistence, 500));
        } else {
            setTimeout(setupPersistence, 500);
        }
        
        // Also run when Gradio updates the DOM
        if (typeof MutationObserver !== 'undefined') {
            const observer = new MutationObserver(() => {
                setTimeout(setupPersistence, 200);
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
    })();
    </script>
    """




# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

def fetch_server_info(base_url: str) -> dict:
    """GET /v1/reap/info"""
    resp = requests.get(f"{base_url}/v1/reap/info", timeout=5)
    resp.raise_for_status()
    return resp.json()


def fetch_stats(base_url: str) -> dict:
    """GET /v1/reap/stats"""
    resp = requests.get(f"{base_url}/v1/reap/stats", timeout=10)
    resp.raise_for_status()
    return resp.json()


def post_save(base_url: str, path: str) -> dict:
    """POST /v1/reap/save"""
    resp = requests.post(
        f"{base_url}/v1/reap/save",
        json={"path": path},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def post_reset(base_url: str) -> dict:
    """POST /v1/reap/reset"""
    resp = requests.post(f"{base_url}/v1/reap/reset", timeout=5)
    resp.raise_for_status()
    return resp.json()


def fetch_steering(base_url: str) -> dict:
    """GET /v1/reap/steer"""
    resp = requests.get(f"{base_url}/v1/reap/steer", timeout=5)
    resp.raise_for_status()
    return resp.json()


def post_steering(base_url: str, config: dict) -> dict:
    """POST /v1/reap/steer"""
    resp = requests.post(
        f"{base_url}/v1/reap/steer",
        json=config,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def delete_steering(base_url: str) -> dict:
    """DELETE /v1/reap/steer"""
    resp = requests.delete(f"{base_url}/v1/reap/steer", timeout=5)
    resp.raise_for_status()
    return resp.json()


def stream_chat(
    base_url: str,
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """Stream chat completion tokens from /v1/chat/completions."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data = decoded[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_freq_heatmap(
    freq: np.ndarray,
    title: str = "Expert Activation Frequency",
) -> plt.Figure:
    """Create a heatmap of expert activation frequencies.

    Args:
        freq: (num_layers, num_experts) activation counts.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    # Use wider, squarer aspect ratio for better browser fit
    num_layers = freq.shape[0]
    width = 14
    height = max(4, min(8, num_layers * 0.1))  # Cap height at 8
    fig, ax = plt.subplots(figsize=(width, height))

    # Use log scale if range is large
    vmax = freq.max()
    vmin = freq.min()
    if vmax > 0 and vmax / max(vmin, 1) > 100:
        im = ax.imshow(
            freq + 1,  # +1 to avoid log(0)
            aspect="auto",
            cmap="YlOrRd",
            norm=LogNorm(vmin=1, vmax=vmax + 1),
            interpolation="nearest",
        )
    else:
        im = ax.imshow(
            freq,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("MoE Layer Index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Activation Count")
    fig.tight_layout()
    return fig


def make_weighted_freq_heatmap(
    weighted_freq: np.ndarray,
    title: str = "Router-Weighted Frequency",
) -> plt.Figure:
    """Create a heatmap of router-weighted frequencies."""
    # Use wider, squarer aspect ratio for better browser fit
    num_layers = weighted_freq.shape[0]
    width = 14
    height = max(4, min(8, num_layers * 0.1))  # Cap height at 8
    fig, ax = plt.subplots(figsize=(width, height))

    im = ax.imshow(
        weighted_freq,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("MoE Layer Index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Weighted Frequency Sum")
    fig.tight_layout()
    return fig


def make_per_layer_bar(
    freq: np.ndarray,
    layer_idx: int,
) -> plt.Figure:
    """Create a horizontal bar chart for one layer's expert frequencies."""
    n_experts = freq.shape[1]
    fig, ax = plt.subplots(figsize=(5, max(5, n_experts * 0.1)))

    values = freq[layer_idx]
    colors = plt.cm.YlOrRd(values / max(values.max(), 1))
    ax.barh(range(n_experts), values, color=colors, height=0.6)
    ax.set_xlabel("Activation Count", fontsize=8)
    ax.set_ylabel("Expert Index", fontsize=8)
    ax.set_title(f"Layer {layer_idx} Expert Activations", fontsize=9)
    
    # Show every 10th tick label on Y-axis
    ax.set_yticks(range(n_experts))
    ax.set_yticklabels([str(i) if i % 10 == 0 else '' for i in range(n_experts)], fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    
    fig.tight_layout()
    return fig


def make_diff_heatmap(
    freq1: np.ndarray,
    freq2: np.ndarray,
    title: str = "Expert Activation Difference",
) -> plt.Figure:
    """Create a heatmap showing the difference between two frequency arrays.

    Uses a diverging colormap (RdBu_r) where:
    - Red = positive differences (file1 > file2)
    - Blue = negative differences (file2 > file1)
    - White = no difference

    Args:
        freq1: (num_layers, num_experts) array from first file.
        freq2: (num_layers, num_experts) array from second file.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    diff = freq1 - freq2
    # Use wider, squarer aspect ratio for better browser fit
    num_layers = diff.shape[0]
    width = 14
    height = max(4, min(8, num_layers * 0.1))  # Cap height at 8
    fig, ax = plt.subplots(figsize=(width, height))

    # Use diverging colormap centered at 0
    max_abs = np.abs(diff).max()
    if max_abs > 0:
        im = ax.imshow(
            diff,
            aspect="auto",
            cmap="RdBu_r",  # Red-blue reversed: red=positive, blue=negative
            vmin=-max_abs,
            vmax=max_abs,
            interpolation="nearest",
        )
    else:
        # All zeros - use a neutral color
        im = ax.imshow(
            diff,
            aspect="auto",
            cmap="gray",
            interpolation="nearest",
        )

    ax.set_xlabel("Expert Index", fontsize=10)
    ax.set_ylabel("MoE Layer Index", fontsize=10)
    ax.set_title(title, fontsize=11)
    
    # Add colorbar with label
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Difference ({title.split('(')[1].split(')')[0] if '(' in title else 'file1 - file2'})",
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app(base_url: str = "http://127.0.0.1:8080") -> gr.Blocks:
    """Build the Gradio frontend application.

    Args:
        base_url: URL of the running REAP server.

    Returns:
        A Gradio Blocks app ready to .launch().
    """

    # JavaScript for localStorage persistence of file paths
    storage_js = """
    () => {
        // Load saved values on page load
        const mergeFields = ['merge_file1', 'merge_file2', 'merge_file3', 'merge_file4'];
        const diffFields = ['diff_file1', 'diff_file2'];
        
        mergeFields.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                const saved = localStorage.getItem('mlx_fun_' + id);
                if (saved) el.value = saved;
                el.addEventListener('input', () => {
                    localStorage.setItem('mlx_fun_' + id, el.value);
                });
            }
        });
        
        diffFields.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                const saved = localStorage.getItem('mlx_fun_' + id);
                if (saved) el.value = saved;
                el.addEventListener('input', () => {
                    localStorage.setItem('mlx_fun_' + id, el.value);
                });
            }
        });
        
        return [];
    }
    """

    with gr.Blocks(
        title="MLX-FUN Dashboard",
        js=storage_js,
    ) as app:

        gr.Markdown("# MLX-FUN Dashboard")
        gr.Markdown(f"Connected to: `{base_url}`")

        # -------------------------------------------------------------------
        # Chat tab
        # -------------------------------------------------------------------
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type a message...",
                    label="Message",
                    scale=8,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=1, maximum=4096, value=512, step=1,
                    label="Max Tokens",
                )
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                    label="Temperature",
                )
                system_prompt = gr.Textbox(
                    value="You are a helpful assistant.",
                    label="System Prompt",
                    lines=1,
                )
            clear_btn = gr.Button("Clear Chat")

            def chat_respond(message, history, system, max_tok, temp):
                if not message.strip():
                    yield history, ""
                    return

                # Build messages list
                messages = []
                if system.strip():
                    messages.append({"role": "system", "content": system})
                for entry in history:
                    messages.append({"role": entry["role"], "content": entry["content"]})
                messages.append({"role": "user", "content": message})

                # Add user message to history
                history = history + [{"role": "user", "content": message}]

                # Stream assistant response
                assistant_text = ""
                history = history + [{"role": "assistant", "content": ""}]
                try:
                    for token in stream_chat(base_url, messages, int(max_tok), temp):
                        assistant_text += token
                        history[-1]["content"] = assistant_text
                        yield history, ""
                except requests.ConnectionError:
                    history[-1]["content"] = "*Error: Cannot connect to server.*"
                    yield history, ""
                except Exception as e:
                    history[-1]["content"] = f"*Error: {e}*"
                    yield history, ""

            send_btn.click(
                chat_respond,
                inputs=[msg_input, chatbot, system_prompt, max_tokens_slider, temp_slider],
                outputs=[chatbot, msg_input],
            )
            msg_input.submit(
                chat_respond,
                inputs=[msg_input, chatbot, system_prompt, max_tokens_slider, temp_slider],
                outputs=[chatbot, msg_input],
            )
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

        # -------------------------------------------------------------------
        # Dashboard tab
        # -------------------------------------------------------------------
        with gr.Tab("Dashboard"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh Stats", variant="primary")
                metric_choice = gr.Radio(
                    ["Frequency", "Weighted Frequency"],
                    value="Frequency",
                    label="Metric",
                )
            with gr.Row():
                info_md = gr.Markdown("Click **Refresh Stats** to load data.")
            with gr.Row():
                with gr.Column(scale=1):
                    heatmap_plot = gr.Plot(label="Expert Activation Heatmap")
                with gr.Column(scale=1):
                    layer_select = gr.Slider(
                        minimum=0, maximum=0, step=1, value=0,
                        label="Layer Index (for bar chart)",
                        interactive=True,
                    )
                    bar_plot = gr.Plot(label="Per-Layer Expert Activations")

            # Store stats in a state variable
            stats_state = gr.State(None)

            def refresh_dashboard(metric):
                try:
                    stats = fetch_stats(base_url)
                except requests.ConnectionError:
                    return (
                        None,
                        "**Error:** Cannot connect to server.",
                        None, None,
                        gr.update(maximum=0, value=0),
                    )
                except Exception as e:
                    return (
                        None,
                        f"**Error:** {e}",
                        None, None,
                        gr.update(maximum=0, value=0),
                    )

                freq = np.array(stats["freq"])
                wfreq = np.array(stats["weighted_freq_sum"])
                n_layers = stats["num_layers"]
                n_experts = stats["num_experts"]
                req_count = stats["request_count"]
                tok_count = stats["token_count"]

                info_text = (
                    f"**Layers:** {n_layers} | "
                    f"**Experts:** {n_experts} | "
                    f"**Requests:** {req_count} | "
                    f"**Tokens:** {tok_count}"
                )

                if metric == "Weighted Frequency":
                    heatmap = make_weighted_freq_heatmap(wfreq)
                else:
                    heatmap = make_freq_heatmap(freq)

                bar = make_per_layer_bar(freq, 0)
                max_layer = max(0, n_layers - 1)

                return (
                    stats,
                    info_text,
                    heatmap,
                    bar,
                    gr.update(maximum=max_layer, value=0),
                )

            refresh_btn.click(
                refresh_dashboard,
                inputs=[metric_choice],
                outputs=[stats_state, info_md, heatmap_plot, bar_plot, layer_select],
            )

            def update_bar(stats, layer_idx):
                if stats is None:
                    return None
                freq = np.array(stats["freq"])
                layer_idx = int(min(layer_idx, freq.shape[0] - 1))
                return make_per_layer_bar(freq, layer_idx)

            layer_select.change(
                update_bar,
                inputs=[stats_state, layer_select],
                outputs=[bar_plot],
            )

            def update_heatmap(stats, metric):
                if stats is None:
                    return None
                if metric == "Weighted Frequency":
                    return make_weighted_freq_heatmap(np.array(stats["weighted_freq_sum"]))
                return make_freq_heatmap(np.array(stats["freq"]))

            metric_choice.change(
                update_heatmap,
                inputs=[stats_state, metric_choice],
                outputs=[heatmap_plot],
            )

        # -------------------------------------------------------------------
        # Steering tab
        # -------------------------------------------------------------------
        with gr.Tab("Steering"):
            gr.Markdown("### Expert Steering Controls")
            gr.Markdown(
                "Configure SteerMoE-style gate logit injection to selectively "
                "activate or deactivate experts during inference."
            )

            with gr.Row():
                steer_status = gr.Markdown("Click **Refresh** to check steering status.")
                steer_refresh_btn = gr.Button("Refresh", scale=0)

            gr.Markdown("---")
            gr.Markdown("#### From Safety Report")
            with gr.Row():
                safety_map_input = gr.Textbox(
                    label="Safety Report Path",
                    placeholder="/path/to/safety_report.json",
                )
                steer_mode_radio = gr.Radio(
                    ["safe", "unsafe"],
                    value="safe",
                    label="Mode",
                )
                apply_safety_btn = gr.Button("Apply Safety Steering", variant="primary")

            gr.Markdown("---")
            gr.Markdown("#### Custom Steering (JSON)")
            custom_steer_json = gr.Code(
                value='{\n  "deactivate": {"0": [1, 2]},\n  "activate": {"0": [5]},\n  "mask_value": -1000000000.0,\n  "boost_value": 10000.0\n}',
                language="json",
                label="Steering Config",
            )
            with gr.Row():
                apply_custom_btn = gr.Button("Apply Custom Steering", variant="primary")
                remove_steer_btn = gr.Button("Remove All Steering", variant="stop")

            steer_result = gr.JSON(label="Result")

            def refresh_steering():
                try:
                    data = fetch_steering(base_url)
                    if data.get("active"):
                        cfg = data["config"]
                        n_deact = sum(len(v) for v in cfg.get("deactivate", {}).values())
                        n_act = sum(len(v) for v in cfg.get("activate", {}).values())
                        return (
                            f"**Steering active.** "
                            f"Deactivating {n_deact} expert-layer pairs, "
                            f"activating {n_act} expert-layer pairs."
                        )
                    return "**No steering active.**"
                except requests.ConnectionError:
                    return "**Error:** Cannot connect to server."
                except Exception as e:
                    return f"**Error:** {e}"

            steer_refresh_btn.click(refresh_steering, outputs=[steer_status])

            def apply_safety_steering(path, mode):
                if not path.strip():
                    return {"error": "Please provide a safety report path."}
                try:
                    return post_steering(base_url, {
                        "safety_map": path.strip(),
                        "mode": mode,
                    })
                except Exception as e:
                    return {"error": str(e)}

            apply_safety_btn.click(
                apply_safety_steering,
                inputs=[safety_map_input, steer_mode_radio],
                outputs=[steer_result],
            )

            def apply_custom_steering(config_json):
                try:
                    config = json.loads(config_json)
                    return post_steering(base_url, config)
                except json.JSONDecodeError as e:
                    return {"error": f"Invalid JSON: {e}"}
                except Exception as e:
                    return {"error": str(e)}

            apply_custom_btn.click(
                apply_custom_steering,
                inputs=[custom_steer_json],
                outputs=[steer_result],
            )

            def remove_steering():
                try:
                    return delete_steering(base_url)
                except Exception as e:
                    return {"error": str(e)}

            remove_steer_btn.click(remove_steering, outputs=[steer_result])

        # -------------------------------------------------------------------
        # Merge Mode Comparison tab
        # -------------------------------------------------------------------
        with gr.Tab("Merge Mode Comparison"):
            gr.Markdown("### Compare Different Merge Modes")
            gr.Markdown(
                "Merge the same input files using different strategies (sum, normalized, max) "
                "and compare the results. Use filters to identify experts for pruning."
            )
            
            # Inject localStorage persistence script
            gr.HTML(get_local_storage_script())

            with gr.Row():
                gr.Markdown("#### Input Files (select .npz files from your computer)")
            
            # Display area for previously selected files (populated by JavaScript)
            saved_files_display = gr.HTML("", elem_id="saved_files_display")
            
            # File inputs using native OS file dialog
            with gr.Row():
                merge_file1 = gr.File(
                    label="File 1 (.npz) - Required",
                    file_types=[".npz"],
                    type="filepath",
                    elem_id="merge_file1",
                )
                merge_file2 = gr.File(
                    label="File 2 (.npz) - Required",
                    file_types=[".npz"],
                    type="filepath",
                    elem_id="merge_file2",
                )
            
            with gr.Row():
                merge_file3 = gr.File(
                    label="File 3 (.npz) - Optional",
                    file_types=[".npz"],
                    type="filepath",
                    elem_id="merge_file3",
                )
                merge_file4 = gr.File(
                    label="File 4 (.npz) - Optional",
                    file_types=[".npz"],
                    type="filepath",
                    elem_id="merge_file4",
                )

            with gr.Row():
                metric_choice_merge = gr.Radio(
                    ["Frequency", "Weighted Frequency", "REAP", "EAN"],
                    value="REAP",
                    label="Metric for Ranking",
                )

            with gr.Row():
                merge_btn = gr.Button("Merge Files", variant="primary")
            
            # Store merged data for filtering
            merge_state = gr.State(None)
            
            # Dynamic Filters Section
            gr.Markdown("---")
            gr.Markdown("#### Selection Mode")
            gr.Markdown(
                "Choose how to select experts for pruning/merging. "
                "**Per-Layer**: Select N experts from each layer independently. "
                "**Model-Wide**: Select N experts globally across all layers."
            )
            
            with gr.Row():
                selection_mode = gr.Radio(
                    ["Per-Layer", "Model-Wide"],
                    value="Per-Layer",
                    label="Selection Mode",
                )
                action_mode = gr.Radio(
                    ["Analyze", "Prune", "Merge"],
                    value="Analyze",
                    label="Action Mode",
                )
            
            gr.Markdown("---")
            gr.Markdown("#### Dynamic Filters")
            gr.Markdown(
                "Filter experts by rank sum thresholds. Lower rank sum = more important expert. "
                "Use N to Prune to select experts for removal."
            )
            
            with gr.Row():
                rank_min_filter = gr.Number(
                    label="Min Rank Sum (leave empty for no min)",
                    value=None,
                    precision=0,
                    elem_id="rank_min_filter",
                )
                rank_max_filter = gr.Number(
                    label="Max Rank Sum (leave empty for no max)",
                    value=None,
                    precision=0,
                    elem_id="rank_max_filter",
                )
                n_prune_filter = gr.Number(
                    label="N to Prune (per-layer or total based on mode)",
                    value=None,
                    precision=0,
                    elem_id="top_n_filter",
                )
            
            with gr.Row():
                apply_filters_btn = gr.Button("Apply Filters", variant="secondary")
                reset_filters_btn = gr.Button("Reset Filters", variant="secondary")
                export_btn = gr.Button("Export Filtered Results", variant="primary")

            with gr.Row():
                filter_info_md = gr.Markdown(
                    "Merge files first, then use filters to narrow down results."
                )

            with gr.Row():
                merge_plot = gr.Plot(label="Summed Ranks (Lower = More Important)")

            with gr.Row():
                merge_json = gr.JSON(label="Statistics")
            
            # Export file outputs
            with gr.Row():
                export_file = gr.File(label="Exported CSV", visible=True)
                export_json_file = gr.File(label="Exported JSON (for CLI)", visible=True)

            def merge_files_rank(f1, f2, f3, f4, metric):
                """Merge input files using rank-based aggregation and return data for filtering.
                
                Args:
                    f1-f4: File paths from gr.File() components. Can be None if not selected.
                    metric: Metric name for ranking.
                """
                from .saliency import SaliencyAccumulator
                from .stats_ops import merge_saliency

                # Collect non-empty file paths (gr.File returns None when no file selected)
                files = [f for f in [f1, f2, f3, f4] if f is not None]

                if len(files) < 2:
                    return (
                        "**Error:** At least 2 files are required.",
                        None,
                        None,
                        None,
                    )

                # Normalize metric names
                metric_map = {
                    "Frequency": "freq",
                    "Weighted Frequency": "weighted_freq",
                    "REAP": "reap",
                    "EAN": "ean",
                }
                metric_key = metric_map.get(metric, "reap")

                try:
                    # Merge using rank-based aggregation
                    merged = merge_saliency(files, metric=metric_key)
                except Exception as e:
                    return (
                        f"**Error merging files:** {e}",
                        None,
                        None,
                        None,
                    )

                # Get summed ranks (stored in freq array)
                summed_ranks = merged.freq.copy()
                
                # Store data for filtering
                merge_data = {
                    "summed_ranks": summed_ranks.tolist(),
                    "num_layers": merged.num_layers,
                    "num_experts": merged.num_experts,
                    "files": files,
                    "metric": metric,
                    "metric_key": metric_key,
                }

                # Create heatmap showing summed ranks using matplotlib
                num_layers, num_experts = summed_ranks.shape
                width = 14
                height = max(4, min(8, num_layers * 0.1))
                fig, ax = plt.subplots(figsize=(width, height))
                
                im = ax.imshow(
                    summed_ranks,
                    aspect="auto",
                    cmap="RdYlGn_r",
                    interpolation="nearest",
                )
                
                ax.set_xlabel("Expert Index")
                ax.set_ylabel("MoE Layer Index")
                ax.set_title(f"Summed Ranks - {metric} (Lower = More Important)")
                fig.colorbar(im, ax=ax, label="Summed Rank (Lower = More Important)")
                fig.tight_layout()

                # Compute statistics
                stats = {
                    "num_files": len(files),
                    "metric": metric_key,
                    "num_layers": merged.num_layers,
                    "num_experts": merged.num_experts,
                    "total_experts": merged.num_layers * merged.num_experts,
                    "rank_sum_min": float(summed_ranks.min()),
                    "rank_sum_max": float(summed_ranks.max()),
                    "rank_sum_mean": float(summed_ranks.mean()),
                    "rank_sum_std": float(summed_ranks.std()),
                    "most_important": [],
                    "least_important": [],
                }

                # Find most important (lowest rank sum) and least important (highest rank sum)
                flat_ranks = summed_ranks.ravel()
                most_important_indices = np.argsort(flat_ranks)[:5]
                least_important_indices = np.argsort(flat_ranks)[-5:][::-1]

                for idx in most_important_indices:
                    layer_idx, expert_idx = np.unravel_index(idx, summed_ranks.shape)
                    stats["most_important"].append({
                        "layer": int(layer_idx),
                        "expert": int(expert_idx),
                        "rank_sum": float(summed_ranks[layer_idx, expert_idx]),
                    })

                for idx in least_important_indices:
                    layer_idx, expert_idx = np.unravel_index(idx, summed_ranks.shape)
                    stats["least_important"].append({
                        "layer": int(layer_idx),
                        "expert": int(expert_idx),
                        "rank_sum": float(summed_ranks[layer_idx, expert_idx]),
                    })

                # Format info text
                info = (
                    f"**Merged {len(files)} files using rank-based aggregation**\n\n"
                    f"**Metric for ranking:** {metric}\n"
                    f"**Dimensions:** {merged.num_layers} layers × {merged.num_experts} experts\n\n"
                    f"**How it works:**\n"
                    f"- Each file ranks experts per-layer (rank 1 = highest score)\n"
                    f"- Ranks are summed across all files\n"
                    f"- **Lower summed rank = more important** (consistently high ranking)\n\n"
                    f"**Statistics:**\n"
                    f"- Rank sum range: [{summed_ranks.min():.0f}, {summed_ranks.max():.0f}]\n"
                    f"- Mean rank sum: {summed_ranks.mean():.1f}\n"
                    f"- Std dev: {summed_ranks.std():.1f}\n\n"
                    f"**Most Important Experts (lowest rank sum):**\n"
                )
                for i, exp in enumerate(stats["most_important"][:3], 1):
                    info += f"  {i}. Layer {exp['layer']}, Expert {exp['expert']}: rank sum = {exp['rank_sum']:.0f}\n"

                return info, fig, stats, merge_data

            merge_btn.click(
                merge_files_rank,
                inputs=[merge_file1, merge_file2, merge_file3, merge_file4, metric_choice_merge],
                outputs=[filter_info_md, merge_plot, merge_json, merge_state],
            )
            
            def apply_filters(merge_data, min_rank, max_rank, n_prune, selection_mode, action_mode):
                """Apply filters to merged data and update visualization.
                
                Supports two selection modes:
                - Per-Layer: Select N experts from each layer independently
                - Model-Wide: Select N experts globally across all layers
                
                Action modes:
                - Analyze: Just visualize the selection
                - Prune: Show what would be removed (for CLI prune command)
                - Merge: Show what would be merged (for CLI merge command)
                """
                if merge_data is None:
                    return (
                        "**No data to filter.** Please merge files first.",
                        None,
                        None,
                    )
                
                summed_ranks = np.array(merge_data["summed_ranks"])
                num_layers = merge_data["num_layers"]
                num_experts = merge_data["num_experts"]
                metric = merge_data["metric"]
                total_experts = num_layers * num_experts
                
                # Create mask for filtering (start with all visible)
                mask = np.ones_like(summed_ranks, dtype=bool)
                filter_desc = []
                prune_candidates = []  # Track which experts would be pruned/merged
                
                # Apply min rank filter
                if min_rank is not None and min_rank > 0:
                    mask &= summed_ranks >= min_rank
                    filter_desc.append(f"rank >= {min_rank}")
                
                # Apply max rank filter
                if max_rank is not None and max_rank > 0:
                    mask &= summed_ranks <= max_rank
                    filter_desc.append(f"rank <= {max_rank}")
                
                # Apply N to prune based on selection mode
                if n_prune is not None and n_prune > 0:
                    if selection_mode == "Model-Wide":
                        # Model-wide: Select N expert INDICES (columns) globally
                        # These columns are pruned from ALL layers
                        n_to_prune_total = int(n_prune)
                        if n_to_prune_total < num_experts:
                            # Compute column-wise scores by summing across all layers
                            column_scores = summed_ranks.sum(axis=0)
                            # Find N columns with highest scores (least important)
                            prune_column_indices = np.argpartition(column_scores, -n_to_prune_total)[-n_to_prune_total:]
                            
                            # Mask out these columns from ALL layers
                            for expert_idx in prune_column_indices:
                                mask[:, expert_idx] = False
                                for layer_idx in range(num_layers):
                                    prune_candidates.append({
                                        "layer": int(layer_idx),
                                        "expert": int(expert_idx),
                                        "rank_sum": float(summed_ranks[layer_idx, expert_idx]),
                                    })
                            filter_desc.append(f"model-wide prune {n_to_prune_total} columns")
                    else:
                        # Per-layer: Select N experts from each layer
                        n_to_prune_per_layer = int(n_prune)
                        if n_to_prune_per_layer < num_experts:
                            for layer_idx in range(num_layers):
                                layer_ranks = summed_ranks[layer_idx]
                                prune_indices = np.argpartition(layer_ranks, -n_to_prune_per_layer)[-n_to_prune_per_layer:]
                                mask[layer_idx, prune_indices] = False
                                for expert_idx in prune_indices:
                                    prune_candidates.append({
                                        "layer": int(layer_idx),
                                        "expert": int(expert_idx),
                                        "rank_sum": float(layer_ranks[expert_idx]),
                                    })
                            filter_desc.append(f"per-layer prune {n_to_prune_per_layer}")
                
                # Count matching experts
                matching_count = mask.sum()
                
                if matching_count == 0:
                    return (
                        "**No experts match the filter criteria.** Try adjusting your filters.",
                        None,
                        {"error": "No matching experts"},
                    )
                
                # Create masked heatmap - preserve original color scale
                masked_ranks = np.where(mask, summed_ranks, np.nan)
                
                # Get original data range for consistent color scaling
                original_vmin = float(summed_ranks.min())
                original_vmax = float(summed_ranks.max())
                
                width = 14
                height = max(4, min(8, num_layers * 0.1))
                fig, ax = plt.subplots(figsize=(width, height))
                
                im = ax.imshow(
                    masked_ranks,
                    aspect="auto",
                    cmap="RdYlGn_r",
                    interpolation="nearest",
                    vmin=original_vmin,
                    vmax=original_vmax,
                )
                
                # Pruned/merged experts are shown as white space (NaN values in masked_ranks)
                # No markers needed - the gap in the heatmap indicates removed experts
                
                ax.set_xlabel("Expert Index")
                ax.set_ylabel("MoE Layer Index")
                title = f"Filtered Summed Ranks - {metric}"
                if filter_desc:
                    title += f" ({', '.join(filter_desc)})"
                ax.set_title(title)
                fig.colorbar(im, ax=ax, label="Summed Rank (Lower = More Important)")
                fig.tight_layout()
                
                # Compute per-layer distribution for model-wide mode
                per_layer_distribution = {}
                if selection_mode == "Model-Wide" and prune_candidates:
                    for c in prune_candidates:
                        layer = c["layer"]
                        per_layer_distribution[str(layer)] = per_layer_distribution.get(str(layer), 0) + 1
                
                # Compute filtered statistics
                filtered_ranks = summed_ranks[mask]
                stats = {
                    "filter_criteria": {
                        "min_rank": min_rank,
                        "max_rank": max_rank,
                        "n_prune": n_prune,
                    },
                    "selection_mode": selection_mode.lower().replace("-", "_"),
                    "action_mode": action_mode.lower(),
                    "matching_experts": int(matching_count),
                    "total_experts": int(total_experts),
                    "experts_per_layer": num_experts,
                    "num_layers": num_layers,
                    "percentage": f"{100 * matching_count / total_experts:.1f}%",
                    "filtered_rank_min": float(filtered_ranks.min()),
                    "filtered_rank_max": float(filtered_ranks.max()),
                    "filtered_rank_mean": float(filtered_ranks.mean()),
                    "filtered_rank_std": float(filtered_ranks.std()),
                    "top_10_most_important": [],
                    "prune_candidates": prune_candidates[:20] if prune_candidates else [],
                    "per_layer_distribution": per_layer_distribution if per_layer_distribution else None,
                }
                
                # Get top 10 most important from filtered results (experts to KEEP)
                filtered_flat_indices = np.where(mask.ravel())[0]
                filtered_ranks_flat = summed_ranks.ravel()[filtered_flat_indices]
                sorted_filtered = np.argsort(filtered_ranks_flat)[:10]
                
                for idx in sorted_filtered:
                    orig_idx = filtered_flat_indices[idx]
                    layer_idx, expert_idx = np.unravel_index(orig_idx, summed_ranks.shape)
                    stats["top_10_most_important"].append({
                        "layer": int(layer_idx),
                        "expert": int(expert_idx),
                        "rank_sum": float(summed_ranks[layer_idx, expert_idx]),
                    })
                
                # Format info text
                action_verb = "merged" if action_mode == "Merge" else "pruned"
                info = (
                    f"**Filtered Results ({action_mode} Mode)**\n\n"
                    f"**Selection Mode:** {selection_mode}\n"
                    f"**Action:** {action_mode}\n"
                    f"**Filters applied:** {', '.join(filter_desc) if filter_desc else 'None'}\n"
                    f"**Matching experts:** {matching_count} / {total_experts} ({100 * matching_count / total_experts:.1f}%)\n\n"
                )
                
                # Add mode-specific info
                if n_prune is not None and n_prune > 0 and prune_candidates:
                    if selection_mode == "Model-Wide":
                        info += (
                            f"**Model-Wide Selection:**\n"
                            f"- Total experts to {action_verb}: {len(prune_candidates)}\n"
                            f"- Showing {matching_count} experts to **KEEP**\n"
                        )
                        if per_layer_distribution:
                            dist_str = ", ".join(f"L{k}:{v}" for k, v in sorted(per_layer_distribution.items())[:10])
                            if len(per_layer_distribution) > 10:
                                dist_str += "..."
                            info += f"- Per-layer distribution: {dist_str}\n"
                        info += "\n"
                    else:
                        info += (
                            f"**Per-Layer Selection:**\n"
                            f"- Experts to {action_verb} per layer: {int(n_prune)}\n"
                            f"- Total {action_verb} candidates: {len(prune_candidates)}\n"
                            f"- Showing {matching_count} experts to **KEEP**\n\n"
                        )
                
                info += (
                    f"**Filtered Statistics:**\n"
                    f"- Rank sum range: [{filtered_ranks.min():.0f}, {filtered_ranks.max():.0f}]\n"
                    f"- Mean rank sum: {filtered_ranks.mean():.1f}\n"
                    f"- Std dev: {filtered_ranks.std():.1f}\n\n"
                    f"**Top 5 Most Important (in filtered set):**\n"
                )
                for i, exp in enumerate(stats["top_10_most_important"][:5], 1):
                    info += f"  {i}. Layer {exp['layer']}, Expert {exp['expert']}: rank sum = {exp['rank_sum']:.0f}\n"
                
                # Show top prune/merge candidates if applicable
                if prune_candidates:
                    sorted_candidates = sorted(prune_candidates, key=lambda x: x["rank_sum"], reverse=True)
                    info += f"\n**Top 5 {action_mode} Candidates (highest rank sum):**\n"
                    for i, exp in enumerate(sorted_candidates[:5], 1):
                        info += f"  {i}. Layer {exp['layer']}, Expert {exp['expert']}: rank sum = {exp['rank_sum']:.0f}\n"
                
                # Add CLI command hint - use --expert-list for Prune/Merge action modes
                if action_mode in ("Prune", "Merge"):
                    if action_mode == "Prune":
                        info += (
                            f"\n**CLI Command (after exporting):**\n"
                            f"```bash\n"
                            f"mlx-fun prune --model ./model --expert-list filtered_experts.json --output ./pruned\n"
                            f"```\n"
                        )
                    elif action_mode == "Merge":
                        info += (
                            f"\n**CLI Command (after exporting):**\n"
                            f"```bash\n"
                            f"mlx-fun merge --model ./model --expert-list filtered_experts.json --dataset calib.jsonl --output ./merged\n"
                            f"```\n"
                        )
                elif n_prune is not None and n_prune > 0:
                    # Analyze mode - show traditional command with --saliency
                    model_wide_flag = "--model-wide " if selection_mode == "Model-Wide" else ""
                    info += (
                        f"\n**CLI Command (traditional):**\n"
                        f"```bash\n"
                        f"mlx-fun prune --model ./model --saliency merged.npz {model_wide_flag}--n-prune {int(n_prune)} --output ./pruned\n"
                        f"```\n"
                    )
                
                return info, fig, stats
            
            apply_filters_btn.click(
                apply_filters,
                inputs=[merge_state, rank_min_filter, rank_max_filter, n_prune_filter, selection_mode, action_mode],
                outputs=[filter_info_md, merge_plot, merge_json],
            )
            
            def reset_filters(merge_data):
                """Reset filters and show original merged data."""
                if merge_data is None:
                    return (
                        "**No data to reset.** Please merge files first.",
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                
                summed_ranks = np.array(merge_data["summed_ranks"])
                num_layers = merge_data["num_layers"]
                num_experts = merge_data["num_experts"]
                metric = merge_data["metric"]
                
                # Create original heatmap
                width = 14
                height = max(4, min(8, num_layers * 0.1))
                fig, ax = plt.subplots(figsize=(width, height))
                
                im = ax.imshow(
                    summed_ranks,
                    aspect="auto",
                    cmap="RdYlGn_r",
                    interpolation="nearest",
                )
                
                ax.set_xlabel("Expert Index")
                ax.set_ylabel("MoE Layer Index")
                ax.set_title(f"Summed Ranks - {metric} (Lower = More Important)")
                fig.colorbar(im, ax=ax, label="Summed Rank (Lower = More Important)")
                fig.tight_layout()
                
                # Compute original statistics
                stats = {
                    "num_files": len(merge_data["files"]),
                    "metric": merge_data["metric_key"],
                    "num_layers": num_layers,
                    "num_experts": num_experts,
                    "total_experts": num_layers * num_experts,
                    "rank_sum_min": float(summed_ranks.min()),
                    "rank_sum_max": float(summed_ranks.max()),
                    "rank_sum_mean": float(summed_ranks.mean()),
                    "rank_sum_std": float(summed_ranks.std()),
                }
                
                info = (
                    f"**Filters reset. Showing all {num_layers * num_experts} experts.**\n\n"
                    f"**Metric:** {metric}\n"
                    f"**Dimensions:** {num_layers} layers × {num_experts} experts\n\n"
                    f"**Statistics:**\n"
                    f"- Rank sum range: [{summed_ranks.min():.0f}, {summed_ranks.max():.0f}]\n"
                    f"- Mean rank sum: {summed_ranks.mean():.1f}\n"
                    f"- Std dev: {summed_ranks.std():.1f}\n"
                )
                
                return info, fig, stats, None, None, None
            
            reset_filters_btn.click(
                reset_filters,
                inputs=[merge_state],
                outputs=[filter_info_md, merge_plot, merge_json, rank_min_filter, rank_max_filter, n_prune_filter],
            )
            
            def export_filtered_results(merge_data, min_rank, max_rank, n_prune, selection_mode, action_mode):
                """Export filtered expert list to CSV and JSON files.
                
                Exports two files:
                - filtered_experts.csv: Human-readable format for inspection
                - filtered_experts.json: Machine-readable keep_map for CLI consumption
                
                Returns:
                    Tuple of (csv_path, json_path) for Gradio download
                """
                if merge_data is None:
                    return None, None
                
                summed_ranks = np.array(merge_data["summed_ranks"])
                num_layers = merge_data["num_layers"]
                num_experts = merge_data["num_experts"]
                total_experts = num_layers * num_experts
                
                # Create mask for filtering (same logic as apply_filters)
                mask = np.ones_like(summed_ranks, dtype=bool)
                prune_candidates = []
                
                if min_rank is not None and min_rank > 0:
                    mask &= summed_ranks >= min_rank
                if max_rank is not None and max_rank > 0:
                    mask &= summed_ranks <= max_rank
                
                # Apply N to prune based on selection mode
                if n_prune is not None and n_prune > 0:
                    if selection_mode == "Model-Wide":
                        # Model-wide: Select N expert INDICES (columns) globally
                        n_to_prune_total = int(n_prune)
                        if n_to_prune_total < num_experts:
                            # Compute column-wise scores by summing across all layers
                            column_scores = summed_ranks.sum(axis=0)
                            # Find N columns with highest scores (least important)
                            prune_column_indices = np.argpartition(column_scores, -n_to_prune_total)[-n_to_prune_total:]
                            # Mask out these columns from ALL layers
                            for expert_idx in prune_column_indices:
                                mask[:, expert_idx] = False
                                for layer_idx in range(num_layers):
                                    prune_candidates.append({
                                        "layer": int(layer_idx),
                                        "expert": int(expert_idx),
                                        "rank_sum": float(summed_ranks[layer_idx, expert_idx]),
                                        "action": action_mode.lower(),
                                    })
                    else:
                        n_to_prune_per_layer = int(n_prune)
                        if n_to_prune_per_layer < num_experts:
                            for layer_idx in range(num_layers):
                                layer_ranks = summed_ranks[layer_idx]
                                prune_indices = np.argpartition(layer_ranks, -n_to_prune_per_layer)[-n_to_prune_per_layer:]
                                mask[layer_idx, prune_indices] = False
                                for expert_idx in prune_indices:
                                    prune_candidates.append({
                                        "layer": int(layer_idx),
                                        "expert": int(expert_idx),
                                        "rank_sum": float(layer_ranks[expert_idx]),
                                        "action": action_mode.lower(),
                                    })
                
                # Collect experts to KEEP
                keep_experts = []
                for layer_idx in range(num_layers):
                    for expert_idx in range(num_experts):
                        if mask[layer_idx, expert_idx]:
                            keep_experts.append({
                                "layer": layer_idx,
                                "expert": expert_idx,
                                "rank_sum": float(summed_ranks[layer_idx, expert_idx]),
                                "action": "keep",
                            })
                
                # Sort by rank sum (most important first)
                keep_experts.sort(key=lambda x: x["rank_sum"])
                prune_candidates.sort(key=lambda x: x["rank_sum"], reverse=True)
                
                # --- Export CSV (human-readable) ---
                all_experts = keep_experts + prune_candidates
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=["layer", "expert", "rank_sum", "action"])
                writer.writeheader()
                writer.writerows(all_experts)
                
                csv_path = "filtered_experts.csv"
                with open(csv_path, "w", newline="") as f:
                    f.write(output.getvalue())
                
                # --- Export JSON (machine-readable keep_map) ---
                # Build keep_map structure
                keep_map = {}
                for layer_idx in range(num_layers):
                    kept = sorted([
                        expert_idx for expert_idx in range(num_experts)
                        if mask[layer_idx, expert_idx]
                    ])
                    keep_map[str(layer_idx)] = kept
                
                json_data = {
                    "version": "1.0",
                    "source": {
                        "files": merge_data.get("files", []),
                        "metric": merge_data.get("metric_key", "reap"),
                        "selection_mode": selection_mode.lower().replace("-", "_"),
                        "action_mode": action_mode.lower()
                    },
                    "filters": {
                        "min_rank": min_rank,
                        "max_rank": max_rank,
                        "n_prune": n_prune
                    },
                    "keep_map": keep_map,
                    "statistics": {
                        "num_layers": num_layers,
                        "num_experts": num_experts,
                        "total_kept": sum(len(v) for v in keep_map.values()),
                        "total_pruned": num_layers * num_experts - sum(len(v) for v in keep_map.values())
                    }
                }
                
                json_path = "filtered_experts.json"
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                
                return csv_path, json_path
            
            export_btn.click(
                export_filtered_results,
                inputs=[merge_state, rank_min_filter, rank_max_filter, n_prune_filter, selection_mode, action_mode],
                outputs=[export_file, export_json_file],
            )

        # -------------------------------------------------------------------
        # Diff Analysis tab (File Comparison)
        # -------------------------------------------------------------------
        with gr.Tab("Diff Analysis"):
            gr.Markdown("### Compare Two Saliency Files")
            gr.Markdown(
                "Visualize differences between two collected saliency files. "
                "Red areas indicate file1 has higher activation, blue indicates file2 has higher activation."
            )

            # File inputs using native OS file dialog
            with gr.Row():
                file1_input = gr.File(
                    label="File 1 (.npz)",
                    file_types=[".npz"],
                    type="filepath",
                    elem_id="diff_file1",
                )
                file2_input = gr.File(
                    label="File 2 (.npz)",
                    file_types=[".npz"],
                    type="filepath",
                    elem_id="diff_file2",
                )

            with gr.Row():
                metric_choice_diff = gr.Radio(
                    ["Frequency", "Weighted Frequency", "REAP", "EAN"],
                    value="Frequency",
                    label="Metric to Compare",
                )
                compare_btn = gr.Button("Compare Files", variant="primary")

            with gr.Row():
                diff_info_md = gr.Markdown("Select files and click **Compare Files** to see differences.")

            with gr.Row():
                diff_heatmap_plot = gr.Plot(label="Difference Heatmap")

            with gr.Row():
                diff_stats_json = gr.JSON(label="Difference Statistics")

            def compare_files(file1, file2, metric):
                """Load two .npz files and compute differences.
                
                Args:
                    file1, file2: File paths from gr.File() components. Can be None.
                    metric: Metric name for comparison.
                """
                from .saliency import SaliencyAccumulator
                from .stats_ops import compute_diff_stats

                # Check that both files are selected (gr.File returns None if not selected)
                if file1 is None or file2 is None:
                    return (
                        "**Error:** Please select both files to compare.",
                        None,
                        {"error": "Both files required"},
                    )

                # Normalize metric names
                metric_map = {
                    "Frequency": "freq",
                    "Weighted Frequency": "weighted_freq",
                    "REAP": "reap",
                    "EAN": "ean",
                }
                metric_key = metric_map.get(metric, "freq")

                # Load files
                try:
                    acc1 = SaliencyAccumulator.load(file1)
                    acc2 = SaliencyAccumulator.load(file2)
                except Exception as e:
                    return (
                        f"**Error loading files:** {e}",
                        None,
                        {"error": str(e)},
                    )

                # Check compatibility
                if acc1.num_layers != acc2.num_layers or acc1.num_experts != acc2.num_experts:
                    return (
                        f"**Error:** Files have incompatible dimensions. "
                        f"File1: ({acc1.num_layers}, {acc1.num_experts}), "
                        f"File2: ({acc2.num_layers}, {acc2.num_experts})",
                        None,
                        {"error": "Incompatible dimensions"},
                    )

                # Compute differences
                try:
                    report = compute_diff_stats(acc1, acc2, metric_key)
                except Exception as e:
                    return (
                        f"**Error computing differences:** {e}",
                        None,
                        {"error": str(e)},
                    )

                # Get arrays for visualization
                if metric_key == "freq":
                    arr1 = acc1.freq
                    arr2 = acc2.freq
                elif metric_key == "weighted_freq":
                    arr1 = acc1.weighted_freq_sum
                    arr2 = acc2.weighted_freq_sum
                elif metric_key == "reap":
                    arr1 = acc1.compute_scores("reap")
                    arr2 = acc2.compute_scores("reap")
                else:  # ean
                    arr1 = acc1.compute_scores("ean")
                    arr2 = acc2.compute_scores("ean")

                # Create heatmap
                heatmap = make_diff_heatmap(
                    arr1,
                    arr2,
                    title=f"Expert Activation Difference ({metric})",
                )

                # Format info text
                info = (
                    f"**Metric:** {metric} | "
                    f"**Dimensions:** {report['num_layers']} layers × {report['num_experts']} experts\n\n"
                    f"**Difference Statistics:**\n"
                    f"- Mean: {report['diff_mean']:.4f}\n"
                    f"- Std: {report['diff_std']:.4f}\n"
                    f"- Range: [{report['diff_min']:.4f}, {report['diff_max']:.4f}]\n\n"
                    f"**Distribution:**\n"
                    f"- Positive (file1 > file2): {report['positive_count']} experts\n"
                    f"- Negative (file2 > file1): {report['negative_count']} experts\n"
                    f"- Zero: {report['zero_count']} experts"
                )

                return info, heatmap, report

            compare_btn.click(
                compare_files,
                inputs=[file1_input, file2_input, metric_choice_diff],
                outputs=[diff_info_md, diff_heatmap_plot, diff_stats_json],
            )

        # -------------------------------------------------------------------
        # Controls tab
        # -------------------------------------------------------------------
        with gr.Tab("Controls"):
            gr.Markdown("### Server Management")

            with gr.Row():
                server_info_btn = gr.Button("Refresh Server Info", variant="primary")
            server_info_json = gr.JSON(label="Server Info")

            server_info_btn.click(
                lambda: _safe_call(lambda: fetch_server_info(base_url)),
                outputs=[server_info_json],
            )

            gr.Markdown("---")
            gr.Markdown("#### Save Saliency Data")
            with gr.Row():
                save_path_input = gr.Textbox(
                    value="reap_saliency.npz",
                    label="Save Path",
                    scale=4,
                )
                save_btn = gr.Button("Save", variant="primary", scale=1)
            save_result = gr.JSON(label="Save Result")

            save_btn.click(
                lambda path: _safe_call(lambda: post_save(base_url, path)),
                inputs=[save_path_input],
                outputs=[save_result],
            )

            gr.Markdown("---")
            gr.Markdown("#### Reset Counters")
            gr.Markdown("*This will clear all accumulated expert activation statistics.*")
            reset_btn = gr.Button("Reset All Stats", variant="stop")
            reset_result = gr.JSON(label="Reset Result")

            reset_btn.click(
                lambda: _safe_call(lambda: post_reset(base_url)),
                outputs=[reset_result],
            )

            gr.Markdown("---")
            gr.Markdown("#### Raw Stats (JSON)")
            raw_stats_btn = gr.Button("Fetch Raw Stats")
            raw_stats_json = gr.JSON(label="Raw Stats")

            raw_stats_btn.click(
                lambda: _safe_call(lambda: fetch_stats(base_url)),
                outputs=[raw_stats_json],
            )

    return app


def _safe_call(fn):
    """Execute fn, catching connection errors."""
    try:
        return fn()
    except requests.ConnectionError:
        return {"error": "Cannot connect to server."}
    except Exception as e:
        return {"error": str(e)}


def launch_frontend(
    server_url: str = "http://127.0.0.1:8080",
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
):
    """Create and launch the Gradio frontend.

    Args:
        server_url: URL of the running REAP server.
        host: Frontend bind address.
        port: Frontend port.
        share: Whether to create a public Gradio share link.
    """
    app = create_app(base_url=server_url)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
    )
