

const API_URL     = "http://localhost:8000";
//const API_URL     = "http://10.236.198.2:5500"; // for live server extension
const SESSION_KEY = "clinic_session_id";   // localStorage key
const MAX_CHARS   = 500;



const messagesEl  = document.getElementById("messages");
const inputEl     = document.getElementById("user-input");
const sendBtnEl   = document.getElementById("send-btn");
const charCountEl = document.getElementById("char-count");
const quickBtns   = document.querySelectorAll(".quick-btn");




function getSessionId() {
  return localStorage.getItem(SESSION_KEY);  // null if not yet set
}

function saveSessionId(id) {
  localStorage.setItem(SESSION_KEY, id);
}


function addBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble " + role;
  bubble.setAttribute("role", "article");

  const inner = document.createElement("div");
  inner.className = "bubble-text";
  inner.textContent = text;

  bubble.appendChild(inner);
  messagesEl.appendChild(bubble);
  scrollToBottom();
  return bubble;
}


let typingBubble = null;

function showTyping() {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  bubble.setAttribute("aria-label", "Assistant is typing");

  const inner = document.createElement("div");
  inner.className = "bubble-text";
  inner.innerHTML = '<div class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></div>';

  bubble.appendChild(inner);
  messagesEl.appendChild(bubble);
  scrollToBottom();
  typingBubble = bubble;
}

function hideTyping() {
  if (typingBubble) {
    typingBubble.remove();
    typingBubble = null;
  }
}




function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}


function setLoading(on) {
  inputEl.disabled   = on;
  sendBtnEl.disabled = on;
  sendBtnEl.textContent = on ? "..." : "Send";
  quickBtns.forEach(btn => { btn.disabled = on; });
}


inputEl.addEventListener("input", () => {
  const remaining = MAX_CHARS - inputEl.value.length;
  if (remaining <= 50) {
    charCountEl.textContent = remaining + " characters remaining";
    charCountEl.classList.add("warning");
  } else {
    charCountEl.textContent = "";
    charCountEl.classList.remove("warning");
  }
});


inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMsg();
  }
});

function quickSend(text) {
  inputEl.value = text;
  sendMsg();
}


async function sendMsg() {

  // 1. read and validate
  const message = inputEl.value.trim();
  if (!message)                  return;   // empty input — do nothing
  if (message.length > MAX_CHARS) return;  // too long — do nothing

  // clear the input immediately — feels responsive
  inputEl.value = "";
  charCountEl.textContent = "";
  charCountEl.classList.remove("warning");

  // 2. show patient's message on the right
  addBubble("user", message);

  // 3. show typing indicator + lock UI
  showTyping();
  setLoading(true);

  try {

   
    const res = await fetch(API_URL + "/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:    message,
        session_id: getSessionId()
      })
    });

    // if the server returned an error status (4xx, 5xx)
    if (!res.ok) {
      throw new Error("Server error: " + res.status);
    }

    // parse the JSON response
    // shape: { response: "...", session_id: "abc-123" }
    const data = await res.json();

    // 5. hide typing indicator
    hideTyping();

    // 6. show the bot's answer on the left
    addBubble("assistant", data.response);

    // 7. save session_id so the next message
    //    continues the same conversation
    if (data.session_id) {
      saveSessionId(data.session_id);
    }

  } catch (err) {

    // network error, Render down, timeout, etc.
    // show a safe fallback — never a blank or crash
    console.error("Chat error:", err);
    hideTyping();
    addBubble(
      "assistant",
      "I'm sorry, I couldn't connect right now.\n" +
      "Please call the clinic directly for assistance."
    );

  } finally {

    // always re-enable the UI so the patient can try again
    setLoading(false);
    inputEl.focus();

  }
}

window.addEventListener("load", () => {
  inputEl.focus();
  scrollToBottom();
});