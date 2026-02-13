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
    fig, ax = plt.subplots(figsize=(14, max(6, freq.shape[0] * 0.25)))

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
    fig, ax = plt.subplots(figsize=(14, max(6, weighted_freq.shape[0] * 0.25)))

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
    """Create a bar chart for one layer's expert frequencies."""
    n_experts = freq.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, n_experts * 0.15), 4))

    values = freq[layer_idx]
    colors = plt.cm.YlOrRd(values / max(values.max(), 1))
    ax.bar(range(n_experts), values, color=colors)
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Activation Count")
    ax.set_title(f"Layer {layer_idx} Expert Activations")
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
        theme=gr.themes.Soft(),
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
                type="messages",
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
                heatmap_plot = gr.Plot(label="Expert Activation Heatmap")
            with gr.Row():
                layer_select = gr.Slider(
                    minimum=0, maximum=0, step=1, value=0,
                    label="Layer Index (for bar chart)",
                    interactive=True,
                )
            with gr.Row():
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
    )
