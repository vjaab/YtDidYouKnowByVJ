---
description: Force a specific topic in the pipeline
---

Use this workflow to generate a video about a specific subject, bypassing the automatic RSS/Web search.

1.  Identify the topic you want to cover.
2.  Run the pipeline with the `--topic` and optionally `--force` flags.

// turbo
3.  Execute the following command, replacing `"YOUR TOPIC HERE"` with your actual topic:
    ```bash
    python main.py --now --topic "YOUR TOPIC HERE" --force
    ```

> [!NOTE]
> The `--topic` flag skips the search phase and tells Gemini to focus purely on your provided text.
> The `--force` flag ensures the video is generated even if you've covered this topic or company recently.
