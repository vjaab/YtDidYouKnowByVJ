import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"
TIMEOUT_SECONDS = 300  # 5 minutes to reply, then auto-select


def _send_message(text, parse_mode="HTML", reply_markup=None):
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    r = requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {})


def _get_updates(offset=None):
    params = {"timeout": 10, "allowed_updates": ["message", "callback_query"]}
    if offset:
        params["offset"] = offset
    r = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("result", [])


def _answer_callback(callback_id, text="✅ Got it!"):
    requests.post(f"{BASE_URL}/answerCallbackQuery",
                  json={"callback_query_id": callback_id, "text": text},
                  timeout=10)


def _edit_message(message_id, text):
    requests.post(f"{BASE_URL}/editMessageText", json={
        "chat_id": CHAT_ID,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML"
    }, timeout=10)


def send_topic_selection(articles):
    """
    Sends a numbered list of top articles to Telegram with inline keyboard buttons.
    Waits up to 5 minutes for the user to tap a number.
    Returns the chosen article dict, or None if timeout (caller falls back to Gemini auto-pick).
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured — skipping topic selection.")
        return None

    # Build message text — top 10 headlines max
    top_articles = articles[:10]
    lines = ["🗞 <b>Today's Tech Headlines</b>\nPick a number to make your video:\n"]
    for i, art in enumerate(top_articles, start=1):
        title   = art.get("title", "Untitled")[:80]
        source  = art.get("source", {}).get("name", "")
        lines.append(f"<b>{i}.</b> {title}\n   <i>— {source}</i>")

    lines.append(f"\n⏳ <i>Auto-selecting in 5 minutes if no reply...</i>")
    message_text = "\n".join(lines)

    # Inline keyboard — 2 buttons per row
    rows = []
    row  = []
    for i in range(1, len(top_articles) + 1):
        row.append({"text": str(i), "callback_data": str(i)})
        if len(row) == 5:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    keyboard = {"inline_keyboard": rows}

    msg = _send_message(message_text, reply_markup=keyboard)
    sent_message_id = msg.get("message_id")
    print(f"Sent {len(top_articles)} topics to Telegram. Waiting for reply...")

    # ── Poll for callback or text reply ──────────────────────────────────────
    deadline = time.time() + TIMEOUT_SECONDS
    last_update_id = None

    # Drain any old pending updates first
    stale = _get_updates()
    if stale:
        last_update_id = stale[-1]["update_id"] + 1

    while time.time() < deadline:
        try:
            updates = _get_updates(offset=last_update_id)
        except Exception as e:
            print(f"Telegram poll error: {e}")
            time.sleep(5)
            continue

        for update in updates:
            last_update_id = update["update_id"] + 1
            choice = None

            # Inline button tap
            if "callback_query" in update:
                cb = update["callback_query"]
                # Only accept if from same chat
                if str(cb["message"]["chat"]["id"]) == str(CHAT_ID):
                    choice = cb["data"].strip()
                    _answer_callback(cb["id"], f"Selected #{choice} ✅")

            # Plain text reply
            elif "message" in update:
                m = update["message"]
                if str(m["chat"]["id"]) == str(CHAT_ID) and "text" in m:
                    choice = m["text"].strip()

            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(top_articles):
                    chosen = top_articles[idx]
                    confirm = (
                        f"✅ <b>Selected #{choice}:</b>\n"
                        f"{chosen.get('title', '')}\n\n"
                        f"🎬 <i>Generating your video now...</i>"
                    )
                    if sent_message_id:
                        _edit_message(sent_message_id, confirm)
                    else:
                        _send_message(confirm)
                    print(f"User chose article #{choice}: {chosen.get('title')}")
                    return chosen

        time.sleep(3)

    # ── Timeout ──────────────────────────────────────────────────────────────
    print("Telegram selection timed out. Falling back to Gemini auto-selection.")
    if sent_message_id:
        _edit_message(sent_message_id,
                      "⏰ <b>Timed out.</b> Gemini will auto-select the best story...")
    return None


def _send_photo(photo_path, caption=""):
    """Send a local image file to Telegram."""
    with open(photo_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
            files={"photo": f},
            timeout=30
        )
    return r.json().get("result", {})


def send_upload_consent(thumbnail_path, title, duration_sec):
    """
    Sends the thumbnail + video details to Telegram and waits for YES/NO.
    Returns True if approved, False if rejected or timed out.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured — auto-approving upload.")
        return True

    mins, secs = divmod(int(duration_sec), 60)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    caption = (
        f"🎬 <b>Video Ready!</b>\n\n"
        f"📌 <b>Title:</b> {title}\n"
        f"⏱ <b>Duration:</b> {duration_str}\n\n"
        f"Upload to YouTube?"
    )

    # Send thumbnail with caption
    try:
        if thumbnail_path and os.path.exists(thumbnail_path):
            _send_photo(thumbnail_path, caption)
        else:
            _send_message(caption)
    except Exception as e:
        print(f"Telegram photo send failed: {e}")
        _send_message(caption)

    # Send YES/NO inline keyboard separately
    keyboard = {"inline_keyboard": [[
        {"text": "✅ YES — Upload Now",  "callback_data": "UPLOAD_YES"},
        {"text": "❌ NO — Skip Upload",  "callback_data": "UPLOAD_NO"},
    ]]}
    msg = _send_message("👆 Tap your choice:", reply_markup=keyboard)
    consent_msg_id = msg.get("message_id")
    print("Upload consent sent to Telegram. Waiting...")

    # Drain stale updates
    deadline = time.time() + 300  # 5 min
    last_update_id = None
    stale = _get_updates()
    if stale:
        last_update_id = stale[-1]["update_id"] + 1

    while time.time() < deadline:
        try:
            updates = _get_updates(offset=last_update_id)
        except Exception as e:
            print(f"Telegram poll error: {e}")
            time.sleep(5)
            continue

        for update in updates:
            last_update_id = update["update_id"] + 1

            if "callback_query" in update:
                cb = update["callback_query"]
                if str(cb["message"]["chat"]["id"]) == str(CHAT_ID):
                    data = cb["data"]
                    if data == "UPLOAD_YES":
                        _answer_callback(cb["id"], "Uploading to YouTube... 🚀")
                        if consent_msg_id:
                            _edit_message(consent_msg_id, "🚀 <b>Uploading to YouTube...</b>")
                        print("User approved upload.")
                        return True
                    elif data == "UPLOAD_NO":
                        _answer_callback(cb["id"], "Upload skipped.")
                        if consent_msg_id:
                            _edit_message(consent_msg_id, "❌ <b>Upload skipped by user.</b>")
                        print("User rejected upload.")
                        return False

            elif "message" in update:
                m = update["message"]
                if str(m["chat"]["id"]) == str(CHAT_ID) and "text" in m:
                    txt = m["text"].strip().lower()
                    if txt in ("yes", "y", "upload"):
                        _send_message("🚀 <b>Uploading to YouTube...</b>")
                        return True
                    elif txt in ("no", "n", "skip"):
                        _send_message("❌ <b>Upload skipped.</b>")
                        return False

        time.sleep(3)

    print("Upload consent timed out — skipping upload.")
    if consent_msg_id:
        _edit_message(consent_msg_id, "⏰ <b>Timed out — upload skipped.</b>")
    return False


def notify_telegram(message, emoji="ℹ️"):
    """Send a plain notification message to Telegram."""
    if BOT_TOKEN and CHAT_ID:
        try:
            _send_message(f"{emoji} {message}")
        except Exception as e:
            print(f"Telegram notify failed: {e}")
