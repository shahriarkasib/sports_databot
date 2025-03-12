// DOM elements
const msgerForm = document.getElementById("msgerForm");
const msgerInput = document.getElementById("textInput");
const msgerChat = document.getElementById("msgerChat");
const suggestedQuestions = document.getElementById("suggestedQuestions");

const progressBox = document.getElementById("progressBox");
const progressSteps = document.getElementById("progressSteps");

// Update these image paths as needed
const BOT_IMG = "{{ url_for('static', filename='images/image.png') }}";
const PERSON_IMG = "{{ url_for('static', filename='images/image.png') }}";

const BOT_NAME = "DataBot";
const PERSON_NAME = "You";

// Auto-resize the text area on input
msgerInput.addEventListener("input", () => {
  msgerInput.style.height = "auto";
  msgerInput.style.height = msgerInput.scrollHeight + "px";
});

// Press Enter (without Shift) to send message
msgerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    msgerForm.dispatchEvent(new Event("submit"));
  }
});

// Allow clicking on suggested questions to auto-send the question
if (suggestedQuestions) {
  suggestedQuestions.addEventListener("click", (event) => {
    if (event.target.matches("li[data-question]")) {
      const questionText = event.target.getAttribute("data-question");
      msgerInput.value = questionText;
      msgerInput.style.height = "auto";
      msgerInput.style.height = msgerInput.scrollHeight + "px";
      msgerForm.dispatchEvent(new Event("submit"));
    }
  });
}

// On form submit, send the user's message via AJAX to /ask
msgerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const userText = msgerInput.value.trim();
  if (!userText) return;
  
  appendMessage(PERSON_NAME, PERSON_IMG, "right", userText);
  msgerInput.value = "";
  msgerInput.style.height = "40px";
  
  // Clear and show progress box
  progressSteps.innerHTML = "";
  progressBox.style.display = "block";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userText }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line in buffer
      
      for (const line of lines) {
        if (!line.trim()) continue;
        
        try {
          const jsonObject = JSON.parse(line);
          if (jsonObject.type === 'progress') {
            // Update progress display with only the current step
            progressSteps.innerHTML = `<li>${jsonObject.data}</li>`;
          } else if (jsonObject.type === 'answer') {
            appendMessage(BOT_NAME, BOT_IMG, "left", jsonObject.data);
            progressBox.style.display = "none";
          }
        } catch (e) {
          console.error('Error parsing JSON:', e);
        }
      }
    }
  } catch (error) {
    console.error('Error:', error);
    progressBox.style.display = "none";
    appendMessage(BOT_NAME, BOT_IMG, "left", "Sorry, there was an error processing your request.");
  }
});


/* =============== Chat Bubble Rendering =============== */

function appendMessage(name, img, side, text) {
  const formatted = parseTextWithTable(text);
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const msgHTML = `
    <div class="msg ${side}-msg">
      <div class="msg-img" style="background-image: url(${img});"></div>
      <div class="msg-bubble">
        <div class="msg-info">
          <div class="msg-info-name">${name}</div>
          <div class="msg-info-time">${time}</div>
        </div>
        <div class="msg-text">${formatted}</div>
      </div>
    </div>
  `;
  $("#msgerChat").append(msgHTML);
  // Scroll the chat to the bottom smoothly
  setTimeout(() => {
    document.querySelector(".msger-wrapper").scrollTo({
      top: document.querySelector(".msger-wrapper").scrollHeight,
      behavior: "smooth",
    });
  }, 100);
}

/* =============== Markdown Table Parsing =============== */

function parseTextWithTable(fullText) {
  const lines = fullText.split("\n").map(l => l.trim());
  let tableLines = [];
  let aboveTable = [];
  let belowTable = [];
  let inTable = false;
  let foundTable = false;
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    if (line.startsWith("|")) {
      inTable = true;
      foundTable = true;
      tableLines.push(line);
    } else {
      if (inTable) {
        inTable = false;
        belowTable.push(line);
      } else {
        if (!foundTable) {
          aboveTable.push(line);
        } else {
          belowTable.push(line);
        }
      }
    }
  }
  let introHTML = formatPlainText(aboveTable.join("\n"));
  let outroHTML = formatPlainText(belowTable.join("\n"));
  let tableHTML = "";
  if (tableLines.length) {
    tableHTML = convertMarkdownToTable(tableLines);
  }
  let finalHTML = "";
  if (introHTML) finalHTML += `<p>${introHTML}</p>`;
  if (tableHTML) finalHTML += tableHTML;
  if (outroHTML) finalHTML += `<p>${outroHTML}</p>`;
  return finalHTML;
}

function convertMarkdownToTable(lines) {
  lines = lines.filter(l => l !== "");
  if (!lines.length) return "";
  let headers = lines[0].split("|").map(h => h.trim()).filter(h => h);
  let start = 1;
  if (lines[1] && lines[1].includes("---")) {
    start = 2;
  }
  let rows = [];
  for (let i = start; i < lines.length; i++) {
    let rowData = lines[i].split("|").map(cell => cell.trim()).filter(c => c);
    rows.push(rowData);
  }
  let html = `<table class="styled-table">
    <thead>
      <tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr>
    </thead>
    <tbody>`;
  rows.forEach(row => {
    html += `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`;
  });
  html += "</tbody></table>";
  return `<div class="msg-text">${html}</div>`;
}

function formatPlainText(text) {
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\n\n/g, "<br><br>");
  text = text.replace(/\n/g, "<br>");
  return text;
}
