(function () {
  "use strict";

  var API_BASE = (window.CLAIM_API_BASE || "").replace(/\/+$/, "");

  var form = document.getElementById("claim-form");
  var input = document.getElementById("text-input");
  var button = document.getElementById("submit-btn");
  var charCount = document.getElementById("char-count");
  var errorBox = document.getElementById("error");
  var resultBox = document.getElementById("result");

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    resultBox.classList.add("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
  }

  function showResult(text, result) {
    // textContent (never innerHTML) so user input can't inject markup.
    document.getElementById("result-text").textContent = '"' + text + '"';

    var classEl = document.getElementById("result-class");
    classEl.textContent = result.is_claim ? "Fact-checkable Claim" : "Not a Claim";
    classEl.className = result.is_claim ? "claim-true" : "claim-false";

    document.getElementById("result-confidence").textContent = result.confidence;
    document.getElementById("result-time").textContent = result.processing_time_ms;
    document.getElementById("result-cached").textContent = result.cached ? "Yes" : "No";

    resultBox.classList.remove("hidden");
    hideError();
  }

  input.addEventListener("input", function () {
    charCount.textContent = input.value.length;
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var text = input.value.trim();
    if (!text) {
      showError("Please enter some text to analyze.");
      return;
    }

    if (!API_BASE || API_BASE.indexOf("example.com") !== -1) {
      showError("Frontend is not configured: set window.CLAIM_API_BASE in config.js to your tunnel hostname.");
      return;
    }

    button.disabled = true;
    button.textContent = "Analyzing...";
    hideError();

    // The Pi runs inference on CPU; give a cold request room to finish.
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 30000);

    fetch(API_BASE + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
      signal: controller.signal
    })
      .then(function (response) {
        if (response.status === 429) {
          throw new Error("Rate limit reached. Wait a moment and try again.");
        }
        if (response.status === 422) {
          throw new Error("Text rejected: it must be between 1 and 1000 characters.");
        }
        if (!response.ok) {
          throw new Error("The API returned an error (HTTP " + response.status + ").");
        }
        return response.json();
      })
      .then(function (result) {
        showResult(text, result);
      })
      .catch(function (err) {
        if (err.name === "AbortError") {
          showError("The request timed out. The Pi may be under load — try again.");
        } else if (err instanceof TypeError) {
          // fetch() rejects with TypeError for network failures *and* for CORS
          // rejections; the browser hides which one for security reasons.
          showError(
            "Could not reach the API at " + API_BASE + ". " +
            "Check that the tunnel is up and that this origin (" + window.location.origin +
            ") is in the API's ALLOWED_ORIGINS."
          );
        } else {
          showError(err.message);
        }
      })
      .finally(function () {
        clearTimeout(timeout);
        button.disabled = false;
        button.textContent = "Analyze Text";
      });
  });
})();
