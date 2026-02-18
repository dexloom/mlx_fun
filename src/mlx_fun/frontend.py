"""Gradio web frontend for the MLX-FUN server.

Connects to a running REAP server and provides:
- Chat interface (via OpenAI-compatible /v1/chat/completions)
- Expert activation dashboard (heatmaps from /v1/reap/stats)
- Steering controls (via /v1/reap/steer)
- Server management (save, reset, info)
"""

import json
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
    fig, ax = plt.subplots(figsize=(10, max(4, freq.shape[0] * 0.2)))

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
    fig, ax = plt.subplots(figsize=(10, max(4, weighted_freq.shape[0] * 0.2)))

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
    fig, ax = plt.subplots(figsize=(10, max(4, diff.shape[0] * 0.2)))

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

    with gr.Blocks(
        title="MLX-FUN Dashboard",
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
                "and compare the results. Select which comparison to view."
            )

            with gr.Row():
                gr.Markdown("#### Input Files")
            with gr.Row():
                merge_file1 = gr.Textbox(
                    label="File 1 Path (.npz)",
                    placeholder="data/reap_saliency_agent_minimax_m25.npz",
                    scale=1,
                )
                merge_file2 = gr.Textbox(
                    label="File 2 Path (.npz)",
                    placeholder="data/reap_saliency_solidity_functions_minimax_m25.npz",
                    scale=1,
                )

            with gr.Row():
                merge_file3 = gr.Textbox(
                    label="File 3 Path (.npz, optional)",
                    placeholder="data/reap_saliency_general_minimax_m25.npz",
                    scale=1,
                )
                merge_file4 = gr.Textbox(
                    label="File 4 Path (.npz, optional)",
                    placeholder="",
                    scale=1,
                )

            with gr.Row():
                metric_choice_merge = gr.Radio(
                    ["Frequency", "Weighted Frequency", "REAP", "EAN"],
                    value="Frequency",
                    label="Metric to Compare",
                )
                comparison_choice = gr.Radio(
                    ["Sum vs Normalized", "Sum vs Max", "Normalized vs Max"],
                    value="Sum vs Normalized",
                    label="Comparison to View",
                )

            with gr.Row():
                compare_modes_btn = gr.Button("Merge and Compare", variant="primary")

            with gr.Row():
                merge_info_md = gr.Markdown("Select input files and click **Merge and Compare** to see results.")

            with gr.Row():
                comparison_plot = gr.Plot(label="Merge Mode Comparison")

            with gr.Row():
                comparison_json = gr.JSON(label="Statistics")

            # Store merged results in state to allow switching comparisons
            merged_results_state = gr.State(None)

            def compare_merge_modes(f1, f2, f3, f4, metric):
                """Merge input files with different modes and compare results."""
                from .saliency import SaliencyAccumulator
                from .stats_ops import merge_saliency, compute_diff_stats

                # Collect non-empty file paths
                files = [f for f in [f1, f2, f3, f4] if f and f.strip()]
                
                if len(files) < 2:
                    return (
                        "**Error:** At least 2 files are required.",
                        None,
                        None,
                        None,  # Return None for comparison_choice state update
                    )

                # Normalize metric names
                metric_map = {
                    "Frequency": "freq",
                    "Weighted Frequency": "weighted_freq",
                    "REAP": "reap",
                    "EAN": "ean",
                }
                metric_key = metric_map.get(metric, "freq")

                try:
                    # Merge with all modes
                    merged_sum = merge_saliency(files, mode="sum")
                    merged_norm = merge_saliency(files, mode="normalized")
                    merged_max = merge_saliency(files, mode="max")
                except Exception as e:
                    return (
                        f"**Error merging files:** {e}",
                        None,
                        None,
                        None,  # Return None for comparison_choice state update
                    )

                # Store all merged results
                merged_results = {
                    "sum": merged_sum,
                    "norm": merged_norm,
                    "max": merged_max,
                }

                # Generate the default comparison (Sum vs Normalized)
                comparison = "Sum vs Normalized"
                info, heatmap, stats = _generate_comparison(
                    merged_results, comparison, metric, metric_key, len(files)
                )

                return info, heatmap, stats, merged_results

            def _generate_comparison(merged_results, comparison, metric_name, metric_key, num_files):
                """Generate a specific comparison based on selection."""
                from .saliency import SaliencyAccumulator
                from .stats_ops import merge_saliency, compute_diff_stats
                
                # Get the two modes to compare
                if comparison == "Sum vs Normalized":
                    mode1, mode2 = "sum", "norm"
                    title = f"Sum Mode - Normalized Mode ({metric_name})"
                elif comparison == "Sum vs Max":
                    mode1, mode2 = "sum", "max"
                    title = f"Sum Mode - Max Mode ({metric_name})"
                else:  # Normalized vs Max
                    mode1, mode2 = "norm", "max"
                    title = f"Normalized Mode - Max Mode ({metric_name})"

                acc1 = merged_results[mode1]
                acc2 = merged_results[mode2]

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
                heatmap = make_diff_heatmap(arr1, arr2, title=title)

                # Compute statistics
                diff_stats = compute_diff_stats(acc1, acc2, metric_key)

                # Format info text
                info = (
                    f"**Merging {num_files} files with different modes...**\n\n"
                    f"**Merge Mode Characteristics:**\n"
                    f"- **Sum mode:** Total samples = {merged_results['sum'].freq.sum():.0f} (larger datasets dominate)\n"
                    f"- **Normalized mode:** Total samples = {merged_results['norm'].freq.sum():.0f} (equal weight per dataset)\n"
                    f"- **Max mode:** Total samples = {merged_results['max'].freq.sum():.0f} (peak activations only)\n\n"
                    f"**Current Comparison:** {comparison}\n"
                    f"**Metric:** {metric_name} | "
                    f"**Dimensions:** {acc1.num_layers} layers × {acc1.num_experts} experts\n\n"
                    f"**Difference Statistics:**\n"
                    f"- Mean: {diff_stats['diff_mean']:.4f}\n"
                    f"- Std: {diff_stats['diff_std']:.4f}\n"
                    f"- Range: [{diff_stats['diff_min']:.4f}, {diff_stats['diff_max']:.4f}]\n\n"
                    f"**Distribution:**\n"
                    f"- Positive ({comparison.split(' vs ')[0]} > {comparison.split(' vs ')[1]}): {diff_stats['positive_count']} experts\n"
                    f"- Negative ({comparison.split(' vs ')[1]} > {comparison.split(' vs ')[0]}): {diff_stats['negative_count']} experts\n"
                    f"- Zero: {diff_stats['zero_count']} experts"
                )

                return info, heatmap, diff_stats

            def switch_comparison(merged_results, comparison, metric):
                """Switch between different comparisons without re-merging."""
                if merged_results is None:
                    return None, None, None

                metric_map = {
                    "Frequency": "freq",
                    "Weighted Frequency": "weighted_freq",
                    "REAP": "reap",
                    "EAN": "ean",
                }
                metric_key = metric_map.get(metric, "freq")
                metric_name = metric

                num_files = 2  # Default, since we don't know the original count
                
                info, heatmap, stats = _generate_comparison(
                    merged_results, comparison, metric_name, metric_key, num_files
                )

                return info, heatmap, stats

            compare_modes_btn.click(
                compare_merge_modes,
                inputs=[merge_file1, merge_file2, merge_file3, merge_file4, metric_choice_merge],
                outputs=[merge_info_md, comparison_plot, comparison_json, merged_results_state],
            )

            comparison_choice.change(
                switch_comparison,
                inputs=[merged_results_state, comparison_choice, metric_choice_merge],
                outputs=[merge_info_md, comparison_plot, comparison_json],
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

            with gr.Row():
                file1_input = gr.Textbox(
                    label="File 1 Path (.npz)",
                    placeholder="data/reap_saliency_agent_minimax_m25.npz",
                    scale=1,
                )
                file2_input = gr.Textbox(
                    label="File 2 Path (.npz)",
                    placeholder="data/reap_saliency_solidity_functions_minimax_m25.npz",
                    scale=1,
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
                """Load two .npz files and compute differences."""
                from .saliency import SaliencyAccumulator
                from .stats_ops import compute_diff_stats

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
